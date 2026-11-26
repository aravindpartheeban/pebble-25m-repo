"""Standardized evaluation suite for static embedding models (NanoBEIR, MNanoBEIR, Matryoshka, CPU speed)."""

from __future__ import annotations

import inspect
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from model2vec import StaticModel
from safetensors import safe_open
from sentence_transformers import SentenceTransformer

from .soup import norm_matched_blend
from .utils import resolve_special_token_ids

try:
    from sentence_transformers.sentence_transformer.evaluation import (
        InformationRetrievalEvaluator,
        NanoBEIREvaluator,
    )
    from sentence_transformers.sentence_transformer.modules import StaticEmbedding
except ImportError:
    from sentence_transformers.evaluation import InformationRetrievalEvaluator, NanoBEIREvaluator
    from sentence_transformers.models import StaticEmbedding


MNANOBEIR_REPOS: dict[str, str] = {
    "en": "sentence-transformers/NanoBEIR-en",
    "ar": "lightonai/NanoBEIR-ar",
    "de": "lightonai/NanoBEIR-de",
    "es": "lightonai/NanoBEIR-es",
    "fr": "lightonai/NanoBEIR-fr",
    "it": "lightonai/NanoBEIR-it",
    "no": "lightonai/NanoBEIR-no",
    "pt": "lightonai/NanoBEIR-pt",
    "sv": "lightonai/NanoBEIR-sv",
    "sr": "Serbian-AI-Society/NanoBEIR-sr",
    "ja": "LiquidAI/NanoBEIR-ja",
    "ko": "LiquidAI/NanoBEIR-ko",
    "vi": "sionic-ai/NanoBEIR-vi",
    "th": "sionic-ai/NanoBEIR-th",
}

NANOBEIR_DATASETS: list[str] = [
    "arguana",
    "climatefever",
    "dbpedia",
    "fever",
    "fiqa2018",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quoraretrieval",
    "scidocs",
    "scifact",
    "touche2020",
]

SPLIT_NAME_MAP: dict[str, str] = {
    "arguana": "NanoArguAna",
    "climatefever": "NanoClimateFEVER",
    "dbpedia": "NanoDBPedia",
    "fever": "NanoFEVER",
    "fiqa2018": "NanoFiQA2018",
    "hotpotqa": "NanoHotpotQA",
    "msmarco": "NanoMSMARCO",
    "nfcorpus": "NanoNFCorpus",
    "nq": "NanoNQ",
    "quoraretrieval": "NanoQuoraRetrieval",
    "scidocs": "NanoSCIDOCS",
    "scifact": "NanoSciFact",
    "touche2020": "NanoTouche2020",
}

BENCH_CORPUS = [
    "How do I rotate production API keys without causing service downtime?",
    "Steps for rotating and revoking live service credentials in production environments.",
    "Why does a Kubernetes pod enter CrashLoopBackOff after a rolling deployment?",
    "Confluence space permissions and page restriction inheritance rules across teams.",
    "What causes long-term bond prices to drop when central bank interest rates rise?",
    "Configuring TLS certificates and HTTP/2 upstream timeouts in an Nginx reverse proxy.",
    "Post-mortem incident review template covering root cause analysis and action items.",
    "Onboarding checklist for setting up local Python virtual environments and Git SSH keys.",
] * 8  # 64 sentences per batch


def parse_model_spec(spec: str) -> tuple[str, int | None]:
    """Parse optional @dim Matryoshka suffix (e.g. 'org/model@256')."""
    if "@" in spec and not Path(spec).exists():
        base, maybe_dim = spec.rsplit("@", 1)
        if maybe_dim.isdigit():
            return base, int(maybe_dim)
    return spec, None


def load_eval_model(spec: str, device: str, truncate_dim: int | None = None) -> tuple[SentenceTransformer, int | None]:
    """Load a local or Hub StaticModel / SentenceTransformer with optional Matryoshka truncation."""
    path_or_id, parsed_dim = parse_model_spec(spec)
    eff_dim = truncate_dim if truncate_dim is not None else parsed_dim
    try:
        se = StaticEmbedding.from_model2vec(path_or_id)
        if eff_dim is not None and se.embedding.weight.shape[1] > eff_dim:
            with torch.no_grad():
                w = se.embedding.weight[:, :eff_dim].detach().cpu().numpy()
            se = StaticEmbedding(se.tokenizer, embedding_weights=w)
            eff_dim = None
        return SentenceTransformer(modules=[se], device=device), eff_dim
    except Exception:
        kwargs: dict = {"device": device}
        if eff_dim is not None:
            kwargs["truncate_dim"] = eff_dim
        return SentenceTransformer(path_or_id, **kwargs), eff_dim


def make_blended_st(s1_path: Path, s2_path: Path, alpha: float, device: str) -> tuple[SentenceTransformer, None]:
    """Construct an in-memory norm-matched blend of Stage-1 and Stage-2 checkpoints."""
    m1 = StaticModel.from_pretrained(str(s1_path))
    m2 = StaticModel.from_pretrained(str(s2_path))
    w1 = np.asarray(m1.embedding, dtype=np.float32)
    w2 = np.asarray(m2.embedding, dtype=np.float32)
    pad_id, _ = resolve_special_token_ids(m1.tokenizer)
    w_blend = norm_matched_blend(w1, w2, alpha=alpha, pad_id=pad_id)
    se = StaticEmbedding(m1.tokenizer, embedding_weights=w_blend)
    return SentenceTransformer(modules=[se], device=device), None


def _read_safetensors_dtype(st_path: Path) -> str:
    try:
        with safe_open(str(st_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if keys:
                dt = str(f.get_slice(keys[0]).get_dtype()).upper()
                return {"F32": "FP32", "F16": "FP16", "BF16": "BF16"}.get(dt, dt)
    except Exception:
        pass
    return "FP32"


def inspect_model_footprint(spec: str) -> tuple[str, int, str, float]:
    """Return (param_str, full_dim, stored_dtype, disk_mb) for a StaticModel checkpoint or HF repo."""
    path_or_id, _ = parse_model_spec(spec)
    p = Path(path_or_id)
    try:
        m = StaticModel.from_pretrained(str(p) if p.exists() else path_or_id)
        w = np.asarray(m.embedding)
        params = w.size
        dim = w.shape[1]
        st_file = (p / "model.safetensors") if p.exists() else None
        if st_file is None:
            try:
                st_file = Path(hf_hub_download(path_or_id, "model.safetensors"))
            except Exception:
                st_file = None
        if st_file is not None and st_file.exists():
            mb = st_file.stat().st_size / (1024 * 1024)
            dtype_str = _read_safetensors_dtype(st_file)
        else:
            mb = w.nbytes / (1024 * 1024)
            dtype_str = "FP32"
        return f"{params / 1e6:.1f}M", dim, dtype_str, mb
    except Exception:
        return "N/A", -1, "N/A", 0.0


def measure_cpu_throughput(spec: str, iters: int = 150) -> float:
    """Measure single-process CPU encoding throughput (docs/sec) on 64-sentence batches."""
    cpu_model, _ = load_eval_model(spec, device="cpu")
    for _ in range(10):
        _ = cpu_model.encode(BENCH_CORPUS, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = cpu_model.encode(BENCH_CORPUS, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    elapsed = time.perf_counter() - t0
    return (len(BENCH_CORPUS) * iters) / max(elapsed, 1e-9)


def build_language_evaluators(
    langs: list[str],
    dataset_names: list[str],
    batch_size: int,
    extra_repos: dict[str, str] | None = None,
) -> dict[str, object]:
    """Pre-build one evaluator per language so datasets are loaded once across models."""
    supports_dataset_id = "dataset_id" in inspect.signature(NanoBEIREvaluator.__init__).parameters
    repo_map = {**MNANOBEIR_REPOS, **(extra_repos or {})}
    evaluators: dict[str, object] = {}

    for lang in langs:
        if lang not in repo_map:
            raise ValueError(f"Unknown language '{lang}'. Available: {sorted(repo_map)}")
        repo_id = repo_map[lang]
        print(f"[Data] Preparing {lang:3s} ({repo_id})...")

        if supports_dataset_id:
            evaluators[lang] = NanoBEIREvaluator(
                dataset_names=dataset_names,
                dataset_id=repo_id,
                batch_size=batch_size,
                mrr_at_k=[10],
                ndcg_at_k=[10],
                precision_recall_at_k=[10, 100],
                write_csv=False,
                show_progress_bar=False,
            )
        else:
            sub_evals: dict[str, InformationRetrievalEvaluator] = {}
            for ds_key in dataset_names:
                split = SPLIT_NAME_MAP[ds_key]
                corpus_ds = load_dataset(repo_id, "corpus", split=split)
                queries_ds = load_dataset(repo_id, "queries", split=split)
                qrels_ds = load_dataset(repo_id, "qrels", split=split)

                corpus = {str(r["_id"]): str(r["text"] or "") for r in corpus_ds}
                queries = {str(r["_id"]): str(r["text"] or "") for r in queries_ds}
                q_col = "query-id" if "query-id" in qrels_ds.column_names else "query_id"
                c_col = "corpus-id" if "corpus-id" in qrels_ds.column_names else "corpus_id"
                rel_docs: dict[str, set[str]] = {}
                for r in qrels_ds:
                    qid, cid = str(r[q_col]), str(r[c_col])
                    if cid in corpus:
                        rel_docs.setdefault(qid, set()).add(cid)
                queries = {qid: txt for qid, txt in queries.items() if qid in rel_docs}

                sub_evals[split] = InformationRetrievalEvaluator(
                    queries=queries,
                    corpus=corpus,
                    relevant_docs=rel_docs,
                    name=split,
                    batch_size=batch_size,
                    mrr_at_k=[10],
                    ndcg_at_k=[10],
                    precision_recall_at_k=[10, 100],
                    write_csv=False,
                    show_progress_bar=False,
                )
            evaluators[lang] = sub_evals

    return evaluators


def run_language_evaluator(
    evaluator_obj: object,
    model: SentenceTransformer,
    truncate_dim: int | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run one language evaluator and return (summary_metrics, per_task_ndcg@10)."""
    per_task_ndcg: dict[str, float] = {}

    if isinstance(evaluator_obj, NanoBEIREvaluator):
        evaluator_obj.truncate_dim = truncate_dim
        raw = evaluator_obj(model)

        def _pick_mean(metric_suffix: str) -> float:
            for k, v in raw.items():
                if "mean" in k.lower() and k.endswith(metric_suffix):
                    return float(v) * 100.0
            vals = [float(v) * 100.0 for k, v in raw.items() if k.endswith(metric_suffix)]
            return float(np.mean(vals)) if vals else 0.0

        for ds_key in NANOBEIR_DATASETS:
            split = SPLIT_NAME_MAP[ds_key]
            for k, v in raw.items():
                if k.startswith(f"{split}_") and k.endswith("ndcg@10"):
                    per_task_ndcg[split] = float(v) * 100.0
                    break

        return (
            {
                "ndcg@10": _pick_mean("ndcg@10"),
                "mrr@10": _pick_mean("mrr@10"),
                "recall@10": _pick_mean("recall@10"),
                "recall@100": _pick_mean("recall@100"),
            },
            per_task_ndcg,
        )

    ndcgs, mrrs, r10s, r100s = [], [], [], []
    for split, sub_ev in evaluator_obj.items():
        sub_ev.truncate_dim = truncate_dim
        res = sub_ev(model)
        ndcg = next(float(v) * 100.0 for k, v in res.items() if k.endswith("ndcg@10"))
        mrr = next(float(v) * 100.0 for k, v in res.items() if k.endswith("mrr@10"))
        r10 = next(float(v) * 100.0 for k, v in res.items() if k.endswith("recall@10"))
        r100 = next(float(v) * 100.0 for k, v in res.items() if k.endswith("recall@100"))
        per_task_ndcg[split] = ndcg
        ndcgs.append(ndcg)
        mrrs.append(mrr)
        r10s.append(r10)
        r100s.append(r100)

    return (
        {
            "ndcg@10": float(np.mean(ndcgs)),
            "mrr@10": float(np.mean(mrrs)),
            "recall@10": float(np.mean(r10s)),
            "recall@100": float(np.mean(r100s)),
        },
        per_task_ndcg,
    )


def run_benchmark(
    models: list[str] | None = None,
    run_dir: Path | None = None,
    sweep_alphas: list[float] | None = None,
    langs: list[str] | None = None,
    mrl_dims: list[int] | None = None,
    measure_speed: bool = True,
    cpu_iters: int = 150,
    batch_size: int = 1024,
    device: str | None = None,
    output_dir: Path = Path("./outputs/eval"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark local checkpoints and/or HF models on NanoBEIR / MNanoBEIR."""
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    eval_langs = langs or ["en"]

    targets: list[tuple[str, str | None, object]] = []
    if models:
        for spec in models:
            label = Path(spec).name if Path(spec).exists() else spec
            targets.append((label, spec, lambda s=spec: load_eval_model(s, resolved_device)))
    elif run_dir is not None:
        for sub in ("stage1_tokenlearn", "stage2a_pretrain", "stage2_contrastive", "final_blended"):
            ckpt = run_dir / sub
            if ckpt.exists():
                targets.append((sub, str(ckpt), lambda p=str(ckpt): load_eval_model(p, resolved_device)))
        if not targets:
            raise FileNotFoundError(f"No checkpoints found under {run_dir}")
    else:
        raise ValueError("Provide either --models or --run-dir.")

    if sweep_alphas and run_dir is not None:
        s1 = run_dir / "stage1_tokenlearn"
        s2 = run_dir / "stage2_contrastive"
        if not (s1.exists() and s2.exists()):
            raise FileNotFoundError("--sweep-alphas requires stage1_tokenlearn and stage2_contrastive in --run-dir")
        for alpha in sweep_alphas:
            targets.append((f"blend_{alpha:.2f}", None, lambda a=alpha: make_blended_st(s1, s2, a, resolved_device)))

    evaluators = build_language_evaluators(eval_langs, NANOBEIR_DATASETS, batch_size=batch_size)

    summary_rows: list[dict] = []
    detailed_rows: list[dict] = []

    for label, spec, loader in targets:
        print(f"\n[Eval] Running {label}...")
        t0 = time.time()
        model, trunc_dim = loader()

        params_str, dim, dtype_str, disk_mb = inspect_model_footprint(spec) if spec else ("blend", -1, "FP32", 0.0)
        cpu_docs_s = measure_cpu_throughput(spec, iters=cpu_iters) if (measure_speed and spec) else None

        per_lang_ndcg: dict[str, float] = {}
        mrrs, r10s, r100s = [], [], []
        for lang in eval_langs:
            summary, task_scores = run_language_evaluator(evaluators[lang], model, trunc_dim)
            per_lang_ndcg[lang] = summary["ndcg@10"]
            mrrs.append(summary["mrr@10"])
            r10s.append(summary["recall@10"])
            r100s.append(summary["recall@100"])
            for task_name, val in task_scores.items():
                detailed_rows.append({"model": label, "lang": lang, "task": task_name, "ndcg@10": val})

        row = {
            "Model": label,
            "Params": params_str,
            "Dim": dim,
            "Weights": f"{dtype_str} ({disk_mb:.1f} MB)" if disk_mb > 0 else "N/A",
            **{f"{lang}_nDCG@10": round(v, 2) for lang, v in per_lang_ndcg.items()},
            "Mean nDCG@10": round(float(np.mean(list(per_lang_ndcg.values()))), 2),
            "MRR@10": round(float(np.mean(mrrs)), 2),
            "Recall@10": round(float(np.mean(r10s)), 2),
            "Recall@100": round(float(np.mean(r100s)), 2),
            "CPU docs/s": f"{cpu_docs_s:,.0f}" if cpu_docs_s is not None else "N/A",
        }
        summary_rows.append(row)
        print(f"[Eval] {label} done in {time.time() - t0:.1f}s | Mean nDCG@10={row['Mean nDCG@10']:.2f}")

    df_summary = pd.DataFrame(summary_rows)
    df_detailed = pd.DataFrame(detailed_rows)

    print("\n" + "=" * 100)
    print("EVALUATION SUMMARY")
    print("=" * 100)
    print(df_summary.to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(output_dir / "summary.csv", index=False)
    df_detailed.round(4).to_csv(output_dir / "per_task.csv", index=False)

    if mrl_dims and models:
        mrl_rows: list[dict] = []
        for spec in models:
            base_label = Path(spec).name if Path(spec).exists() else spec
            for dim in mrl_dims:
                m, tdim = load_eval_model(spec, resolved_device, truncate_dim=dim)
                res, _ = run_language_evaluator(evaluators[eval_langs[0]], m, tdim)
                mrl_rows.append(
                    {
                        "Model": base_label,
                        "Dim": dim,
                        "nDCG@10": round(res["ndcg@10"], 2),
                        "MRR@10": round(res["mrr@10"], 2),
                        "Recall@10": round(res["recall@10"], 2),
                    }
                )
        df_mrl = pd.DataFrame(mrl_rows)
        print("\n" + "=" * 72)
        print(f"MATRYOSHKA TRUNCATION ({eval_langs[0]})")
        print("=" * 72)
        print(df_mrl.to_string(index=False))
        df_mrl.to_csv(output_dir / "mrl_dims.csv", index=False)

    return df_summary, df_detailed
