"""Teacher featurization: encode text corpora into compact FP16 Parquet shards."""

from __future__ import annotations

import hashlib
import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


@dataclass
class FeaturizeConfig:
    model_id: str = "MongoDB/voyage-4-nano"
    trust_remote_code: bool = True
    dim: int | None = 1024
    max_seq_length: int = 512
    encode_mode: str = "auto"  # auto | document | encode

    # Corpus source: Hugging Face dataset OR local files
    dataset: str = "ccde/fineweb-edu-sample-500K"
    dataset_config: str | None = None
    dataset_split: str = "train"
    text_column: str = "text"
    local_files: list[str] | None = None

    output_dir: Path = Path("./data/teacher_features")
    target_rows: int = 500_000
    rows_per_file: int = 100_000
    batch_size: int = 64
    min_words: int = 25
    max_words: int = 320
    shard_id: int = 0
    num_shards: int = 1
    device: str | None = None
    push_to_hub: str = ""
    smoke_test: bool = False


def load_teacher_model(
    model_id: str,
    dim: int | None,
    max_seq_length: int,
    device: str,
    trust_remote_code: bool = True,
) -> SentenceTransformer:
    """Load a SentenceTransformer teacher with automatic dtype selection."""
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device.startswith("cuda") else torch.float32)

    from transformers import AutoModel

    sig = inspect.signature(AutoModel.from_pretrained).parameters
    dtype_key = "dtype" if "dtype" in sig else "torch_dtype"
    model_kwargs: dict = {dtype_key: dtype}
    if device.startswith("cuda"):
        model_kwargs["attn_implementation"] = "sdpa"

    print(
        f"[Teacher] Loading {model_id} | truncate_dim={dim} | max_seq={max_seq_length} | "
        f"{dtype_key}={dtype} | device={device}"
    )
    st_kwargs: dict = {
        "trust_remote_code": trust_remote_code,
        "device": device,
        "model_kwargs": model_kwargs,
    }
    if dim is not None and dim > 0:
        st_kwargs["truncate_dim"] = dim

    model = SentenceTransformer(model_id, **st_kwargs)
    model.max_seq_length = max_seq_length
    return model


def clean_and_clip(text: str, min_words: int, max_words: int) -> str | None:
    """Normalize whitespace, filter short passages, and clip to max_words."""
    if not text:
        return None
    normalized = " ".join(text.split())
    words = normalized.split(" ")
    if len(words) < min_words:
        return None
    if len(words) > max_words:
        normalized = " ".join(words[:max_words])
    return normalized


def _iter_raw_rows(cfg: FeaturizeConfig) -> Iterator[dict]:
    if cfg.local_files:
        paths = [Path(p) for p in cfg.local_files]
        ext = paths[0].suffix.lower()
        fmt = "parquet" if ext == ".parquet" else ("json" if ext in {".json", ".jsonl"} else "text")
        ds = load_dataset(fmt, data_files=[str(p) for p in paths], split="train", streaming=True)
        yield from ds
        return

    load_kwargs: dict = {"split": cfg.dataset_split, "streaming": True}
    if cfg.dataset_config:
        ds = load_dataset(cfg.dataset, name=cfg.dataset_config, **load_kwargs)
    else:
        ds = load_dataset(cfg.dataset, **load_kwargs)
    yield from ds


def iter_texts(cfg: FeaturizeConfig, target_rows: int) -> Iterator[str]:
    """Stream, shard, deduplicate, and clean texts from the configured corpus."""
    seen_hashes: set[bytes] = set()
    emitted = 0

    for raw_idx, row in enumerate(_iter_raw_rows(cfg)):
        if emitted >= target_rows:
            break
        if raw_idx % cfg.num_shards != cfg.shard_id:
            continue

        txt = clean_and_clip(str(row.get(cfg.text_column, "") or ""), cfg.min_words, cfg.max_words)
        if txt is None:
            continue

        digest = hashlib.blake2b(txt[:256].encode("utf-8", errors="ignore"), digest_size=8).digest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        emitted += 1
        yield txt


def write_fp16_parquet(out_path: Path, texts: list[str], embs: np.ndarray) -> None:
    """Save (text, embedding) rows as zstd-compressed FP16 FixedSizeList Parquet."""
    embs_f16 = np.ascontiguousarray(embs, dtype=np.float16)
    dim = embs_f16.shape[1]
    emb_arr = pa.FixedSizeListArray.from_arrays(pa.array(embs_f16.reshape(-1), type=pa.float16()), dim)
    table = pa.table({"text": pa.array(texts, type=pa.string()), "embedding": emb_arr})
    pq.write_table(table, out_path, compression="zstd")


def encode_documents(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    encode_mode: str = "auto",
) -> np.ndarray:
    """Encode a batch of documents using the teacher model and validate finiteness."""
    use_doc_method = encode_mode == "document" or (encode_mode == "auto" and hasattr(model, "encode_document"))
    fn = model.encode_document if use_doc_method else model.encode
    vecs = fn(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vecs = np.asarray(vecs, dtype=np.float32)
    if not np.isfinite(vecs).all():
        raise RuntimeError("Non-finite embeddings detected; switch dtype to float32 or check batch size.")
    return vecs


def run_featurize(cfg: FeaturizeConfig) -> list[Path]:
    """Run teacher feature extraction and return paths to written Parquet shards."""
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    target_rows = 500 if cfg.smoke_test else cfg.target_rows
    rows_per_file = 500 if cfg.smoke_test else cfg.rows_per_file

    model = load_teacher_model(
        model_id=cfg.model_id,
        dim=cfg.dim,
        max_seq_length=cfg.max_seq_length,
        device=device,
        trust_remote_code=cfg.trust_remote_code,
    )

    hf_api = None
    if cfg.push_to_hub and not cfg.smoke_test:
        from huggingface_hub import HfApi

        hf_api = HfApi()
        hf_api.create_repo(cfg.push_to_hub, repo_type="dataset", private=True, exist_ok=True)

    buf_texts: list[str] = []
    buf_embs: list[np.ndarray] = []
    chunk_texts: list[str] = []
    written_files: list[Path] = []
    file_idx = 0
    total_done = 0
    t0 = time.time()

    def _flush_shard() -> None:
        nonlocal file_idx, chunk_texts, buf_embs
        out_file = cfg.output_dir / f"shard{cfg.shard_id}_{file_idx:04d}.parquet"
        write_fp16_parquet(out_file, chunk_texts, np.concatenate(buf_embs, axis=0))
        written_files.append(out_file)
        if hf_api is not None:
            hf_api.upload_file(
                path_or_fileobj=str(out_file),
                path_in_repo=f"data/{out_file.name}",
                repo_id=cfg.push_to_hub,
                repo_type="dataset",
            )
        file_idx += 1
        chunk_texts = []
        buf_embs = []

    pbar = tqdm(total=target_rows, desc=f"Featurize shard {cfg.shard_id}/{cfg.num_shards}")
    for txt in iter_texts(cfg, target_rows=target_rows):
        buf_texts.append(txt)
        if len(buf_texts) >= cfg.batch_size * 4:
            embs = encode_documents(model, buf_texts, cfg.batch_size, cfg.encode_mode)
            chunk_texts.extend(buf_texts)
            buf_embs.append(embs)
            total_done += len(buf_texts)
            pbar.update(len(buf_texts))
            buf_texts = []
            if len(chunk_texts) >= rows_per_file:
                _flush_shard()

    if buf_texts:
        embs = encode_documents(model, buf_texts, cfg.batch_size, cfg.encode_mode)
        chunk_texts.extend(buf_texts)
        buf_embs.append(embs)
        total_done += len(buf_texts)
        pbar.update(len(buf_texts))

    if chunk_texts:
        _flush_shard()

    pbar.close()
    dt = max(time.time() - t0, 1e-6)
    print(f"[Done] Encoded {total_done:,} passages in {dt:.1f}s ({total_done / dt:.1f} docs/s) -> {cfg.output_dir}")
    return written_files
