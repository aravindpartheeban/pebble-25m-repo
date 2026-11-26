"""Dataset loaders for Stage 2a (positive pairs) and Stage 2b (hard negatives + optional scores)."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
from datasets import DatasetDict, load_dataset


# ---------------------------------------------------------------------------
# Stage 2a: Streaming Temperature-Balanced Multi-Domain Pairs
# ---------------------------------------------------------------------------


def discover_pretrain_subsets(pairs_dir: Path) -> dict[str, list[Path]]:
    """Group root or nested Parquet shards by logical subset name, ignoring .cache dirs."""
    subsets: dict[str, list[Path]] = {}
    for p in sorted(pairs_dir.rglob("*.parquet")):
        if ".cache" in p.parts:
            continue
        if p.parent != pairs_dir:
            name = p.parent.name
        else:
            name = re.sub(r"-\d+-of-\d+$", "", p.stem)
        subsets.setdefault(name, []).append(p)
    return subsets


class StreamingPairSubset:
    """Cycles endlessly over one subset's Parquet files while holding only one batch in RAM."""

    def __init__(
        self,
        name: str,
        files: list[Path],
        batch_size: int,
        seed: int,
        q_col: str | None = None,
        d_col: str | None = None,
    ) -> None:
        self.name = name
        self.files = list(files)
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.num_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in self.files)

        pf0 = pq.ParquetFile(self.files[0])
        cols = set(pf0.schema_arrow.names)
        self.q_col = q_col if (q_col and q_col in cols) else next(
            (c for c in ("query", "anchor", "question", "text") if c in cols), None
        )
        self.d_col = d_col if (d_col and d_col in cols) else next(
            (c for c in ("document", "positive", "pos", "passage", "answer") if c in cols), None
        )
        self.valid = self.q_col is not None and self.d_col is not None and self.num_rows >= 4
        self._iter: Iterator[tuple[list[str], list[str]]] | None = None

    def _batch_generator(self) -> Iterator[tuple[list[str], list[str]]]:
        assert self.q_col is not None and self.d_col is not None
        while True:
            order = list(self.files)
            self.rng.shuffle(order)
            for fp in order:
                pf = pq.ParquetFile(fp)
                for batch in pf.iter_batches(batch_size=self.batch_size, columns=[self.q_col, self.d_col]):
                    qs = batch.column(self.q_col).to_pylist()
                    ds = batch.column(self.d_col).to_pylist()
                    seen_q: set[str] = set()
                    seen_d: set[str] = set()
                    clean_q: list[str] = []
                    clean_d: list[str] = []
                    for q, d in zip(qs, ds):
                        q_s = (q or "").strip()[:1500]
                        d_s = (d or "").strip()[:2500]
                        if q_s and d_s and q_s != d_s and q_s not in seen_q and d_s not in seen_d:
                            seen_q.add(q_s)
                            seen_d.add(d_s)
                            clean_q.append(q_s)
                            clean_d.append(d_s)
                    if len(clean_q) >= 4:
                        yield clean_q, clean_d

    def next_batch(self) -> tuple[list[str], list[str]]:
        if self._iter is None:
            self._iter = self._batch_generator()
        return next(self._iter)


# ---------------------------------------------------------------------------
# Stage 2b: ID-Indexed Hard-Negative Dataset (LightOn Schema or Generic Triplets)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IndexedKDExample:
    query: str
    pos_id: int
    neg_ids: tuple[int, ...]
    rerank_scores: tuple[float, ...]


def _calibrate_teacher_temp(sample_rscores: list[float]) -> float:
    if not sample_rscores:
        return 1.0
    arr = np.asarray(sample_rscores, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 1.0
    s_min, s_max, s_std = float(arr.min()), float(arr.max()), float(arr.std())
    return 0.1 if (s_min >= -0.01 and s_max <= 1.01) else max(1.0, s_std * 0.5)


def _load_lighton_indexed(
    source: str,
    max_split_rows: int,
    max_negs: int,
    max_chars: int,
    val_per_split: int,
    smoke_test: bool,
    seed: int,
) -> tuple[dict[str, dict[int, str]], dict[str, list[IndexedKDExample]], dict[str, list[IndexedKDExample]], float]:
    """Load LightOn-style 3-config datasets (scores, documents, queries) with pruned doc maps."""
    print(f"[Stage 2b] Loading LightOn-schema dataset (ID-indexed) from '{source}'...")
    scores_dd = load_dataset(source, "scores")
    docs_dd = load_dataset(source, "documents")
    queries_dd = load_dataset(source, "queries")

    doc_maps: dict[str, dict[int, str]] = {}
    train_splits: dict[str, list[IndexedKDExample]] = {}
    val_splits: dict[str, list[IndexedKDExample]] = {}
    sample_rscores: list[float] = []
    rng = random.Random(seed)
    cap = 256 if smoke_test else max_split_rows

    for split_name in sorted(scores_dd.keys()):
        if split_name not in docs_dd or split_name not in queries_dd:
            continue

        q_ds = queries_dd[split_name]
        d_ds = docs_dd[split_name]
        s_ds = scores_dd[split_name]

        q_map = {int(qid): (txt or "").strip()[:max_chars] for qid, txt in zip(q_ds["query_id"], q_ds["query"])}
        full_d_map = {
            int(did): (txt or "").strip()[:max_chars] for did, txt in zip(d_ds["document_id"], d_ds["document"])
        }

        score_col = (
            "rerank_scores"
            if "rerank_scores" in s_ds.column_names
            else ("scores" if "scores" in s_ds.column_names else None)
        )

        qids = s_ds["query_id"]
        doc_id_lists = s_ds["document_ids"]
        rscore_lists = s_ds[score_col] if score_col else [None] * len(qids)

        examples: list[IndexedKDExample] = []
        used_doc_ids: set[int] = set()

        for idx in range(len(qids)):
            q_txt = q_map.get(int(qids[idx]), "")
            dids = doc_id_lists[idx]
            if not q_txt or not dids or len(dids) < 2:
                continue
            pos_id = int(dids[0])
            pos_txt = full_d_map.get(pos_id, "")
            if not pos_txt:
                continue

            raw_rs = rscore_lists[idx] if rscore_lists[idx] is not None else None
            neg_ids: list[int] = []
            aligned_scores: list[float] = []
            if raw_rs is not None and len(raw_rs) == len(dids):
                aligned_scores.append(float(raw_rs[0]))

            for j in range(1, min(len(dids), 1 + max_negs)):
                nid = int(dids[j])
                nt = full_d_map.get(nid, "")
                if nt and nid != pos_id and nt != pos_txt:
                    neg_ids.append(nid)
                    if aligned_scores:
                        aligned_scores.append(float(raw_rs[j]))

            if not neg_ids:
                continue
            if len(aligned_scores) != 1 + len(neg_ids):
                aligned_scores = []
            elif len(sample_rscores) < 2000:
                sample_rscores.extend(aligned_scores)

            used_doc_ids.add(pos_id)
            used_doc_ids.update(neg_ids)
            examples.append(
                IndexedKDExample(
                    query=q_txt,
                    pos_id=pos_id,
                    neg_ids=tuple(neg_ids),
                    rerank_scores=tuple(aligned_scores),
                )
            )
            if cap > 0 and len(examples) >= cap + val_per_split:
                break

        if len(examples) < 4:
            continue

        doc_maps[split_name] = {did: full_d_map[did] for did in used_doc_ids}
        del full_d_map, q_map

        rng.shuffle(examples)
        nv = min(val_per_split, max(1, len(examples) // 20))
        val_splits[split_name] = examples[:nv]
        train_splits[split_name] = examples[nv:]
        print(
            f"  split={split_name:18s} | train={len(train_splits[split_name]):7,d} "
            f"| val={nv:4d} | unique_docs={len(doc_maps[split_name]):8,d}"
        )

    if not train_splits:
        raise RuntimeError(f"Loaded 0 splits from '{source}'")

    auto_t = _calibrate_teacher_temp(sample_rscores)
    total_tr = sum(len(v) for v in train_splits.values())
    print(f"[Stage 2b] Total train queries={total_tr:,} across {len(train_splits)} splits | auto T_teacher={auto_t:.2f}")
    return doc_maps, train_splits, val_splits, auto_t


def _load_generic_triplets(
    source: str,
    dataset_config: str | None,
    max_split_rows: int,
    max_negs: int,
    max_chars: int,
    val_per_split: int,
    smoke_test: bool,
    seed: int,
) -> tuple[dict[str, dict[int, str]], dict[str, list[IndexedKDExample]], dict[str, list[IndexedKDExample]], float]:
    """
    Load standard Parquet/JSONL/HF hard-negative datasets with columns:
      - query (or anchor/question)
      - positive (or pos/document)
      - negatives (list[str], or single negative string column)
      - optional rerank_scores / scores (aligned with [positive, *negatives])
    """
    print(f"[Stage 2b] Loading generic triplet dataset from '{source}'...")
    src_path = Path(source)
    if src_path.exists():
        files = sorted(src_path.rglob("*.parquet")) if src_path.is_dir() else [src_path]
        ext = files[0].suffix.lower()
        fmt = "parquet" if ext == ".parquet" else "json"
        ds_loaded = load_dataset(fmt, data_files=[str(f) for f in files])
    elif dataset_config:
        ds_loaded = load_dataset(source, name=dataset_config)
    else:
        ds_loaded = load_dataset(source)

    if not isinstance(ds_loaded, DatasetDict):
        ds_loaded = DatasetDict({"train": ds_loaded})

    doc_maps: dict[str, dict[int, str]] = {}
    train_splits: dict[str, list[IndexedKDExample]] = {}
    val_splits: dict[str, list[IndexedKDExample]] = {}
    sample_rscores: list[float] = []
    rng = random.Random(seed)
    cap = 256 if smoke_test else max_split_rows

    for split_name in sorted(ds_loaded.keys()):
        ds = ds_loaded[split_name]
        cols = set(ds.column_names)
        q_col = next((c for c in ("query", "anchor", "question") if c in cols), None)
        p_col = next((c for c in ("positive", "pos", "document", "passage") if c in cols), None)
        n_col = next((c for c in ("negatives", "negative", "neg", "hard_negatives") if c in cols), None)
        s_col = next((c for c in ("rerank_scores", "scores") if c in cols), None)
        if not (q_col and p_col and n_col):
            continue

        text_to_id: dict[str, int] = {}
        dmap: dict[int, str] = {}

        def _intern(doc: str) -> int:
            did = text_to_id.get(doc)
            if did is None:
                did = len(text_to_id)
                text_to_id[doc] = did
                dmap[did] = doc
            return did

        examples: list[IndexedKDExample] = []
        for row in ds:
            q_txt = str(row[q_col] or "").strip()[:max_chars]
            pos_txt = str(row[p_col] or "").strip()[:max_chars]
            raw_negs = row[n_col]
            if isinstance(raw_negs, str):
                raw_negs = [raw_negs]
            if not q_txt or not pos_txt or not raw_negs:
                continue

            raw_rs = row[s_col] if s_col and row[s_col] is not None else None
            aligned_scores: list[float] = []
            if raw_rs is not None and len(raw_rs) == 1 + len(raw_negs):
                aligned_scores.append(float(raw_rs[0]))

            pos_id = _intern(pos_txt)
            neg_ids: list[int] = []
            for j, nt_raw in enumerate(raw_negs[:max_negs]):
                nt = str(nt_raw or "").strip()[:max_chars]
                if nt and nt != pos_txt:
                    neg_ids.append(_intern(nt))
                    if aligned_scores:
                        aligned_scores.append(float(raw_rs[1 + j]))

            if not neg_ids:
                continue
            if len(aligned_scores) != 1 + len(neg_ids):
                aligned_scores = []
            elif len(sample_rscores) < 2000:
                sample_rscores.extend(aligned_scores)

            examples.append(
                IndexedKDExample(
                    query=q_txt,
                    pos_id=pos_id,
                    neg_ids=tuple(neg_ids),
                    rerank_scores=tuple(aligned_scores),
                )
            )
            if cap > 0 and len(examples) >= cap + val_per_split:
                break

        if len(examples) < 4:
            continue

        doc_maps[split_name] = dmap
        rng.shuffle(examples)
        nv = min(val_per_split, max(1, len(examples) // 20))
        val_splits[split_name] = examples[:nv]
        train_splits[split_name] = examples[nv:]
        print(
            f"  split={split_name:18s} | train={len(train_splits[split_name]):7,d} "
            f"| val={nv:4d} | unique_docs={len(dmap):8,d}"
        )

    if not train_splits:
        raise RuntimeError(f"Loaded 0 valid triplet splits from '{source}'")

    auto_t = _calibrate_teacher_temp(sample_rscores)
    return doc_maps, train_splits, val_splits, auto_t


def load_hard_neg_dataset(
    source: str,
    dataset_format: str = "auto",
    dataset_config: str | None = None,
    max_split_rows: int = 0,
    max_negs: int = 10,
    max_chars: int = 2500,
    val_per_split: int = 64,
    smoke_test: bool = False,
    seed: int = 42,
) -> tuple[dict[str, dict[int, str]], dict[str, list[IndexedKDExample]], dict[str, list[IndexedKDExample]], float]:
    """Auto-detect LightOn 3-config format vs. generic hard-negative triplets."""
    if dataset_format == "lighton":
        return _load_lighton_indexed(source, max_split_rows, max_negs, max_chars, val_per_split, smoke_test, seed)
    if dataset_format == "triplet":
        return _load_generic_triplets(
            source, dataset_config, max_split_rows, max_negs, max_chars, val_per_split, smoke_test, seed
        )

    # Auto-detect LightOn layout if local directory contains scores/documents/queries subdirs
    src_path = Path(source)
    if src_path.is_dir() and all((src_path / sub).exists() for sub in ("scores", "documents", "queries")):
        return _load_lighton_indexed(source, max_split_rows, max_negs, max_chars, val_per_split, smoke_test, seed)

    # Try generic triplets first if dataset_config is given or local parquet/jsonl; otherwise try LightOn then generic
    if src_path.exists() or dataset_config is not None:
        return _load_generic_triplets(
            source, dataset_config, max_split_rows, max_negs, max_chars, val_per_split, smoke_test, seed
        )

    try:
        return _load_lighton_indexed(source, max_split_rows, max_negs, max_chars, val_per_split, smoke_test, seed)
    except Exception:
        return _load_generic_triplets(
            source, dataset_config, max_split_rows, max_negs, max_chars, val_per_split, smoke_test, seed
        )
