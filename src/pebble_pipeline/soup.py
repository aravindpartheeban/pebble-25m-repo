"""Stage 3: Norm-matched 2-way blending and N-way model souping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from model2vec import StaticModel
from sentence_transformers import SentenceTransformer

from .utils import build_static_model, resolve_special_token_ids

try:
    from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator
    from sentence_transformers.sentence_transformer.modules import StaticEmbedding
except ImportError:
    from sentence_transformers.evaluation import NanoBEIREvaluator
    from sentence_transformers.models import StaticEmbedding


def load_weights(path: Path) -> tuple[np.ndarray, object, int | None]:
    """Load embedding weights (float32), tokenizer, and max_length from a StaticModel checkpoint."""
    model = StaticModel.from_pretrained(str(path))
    max_length = getattr(model, "max_length", 512)
    return np.asarray(model.embedding, dtype=np.float32), model.tokenizer, max_length


def norm_matched_blend(
    w1: np.ndarray,
    w2: np.ndarray,
    alpha: float,
    pad_id: int | None = 0,
) -> np.ndarray:
    """Blend two checkpoints after rescaling w2 to match w1's median non-pad token norm."""
    mask = np.ones(w1.shape[0], dtype=bool)
    if pad_id is not None:
        mask[pad_id] = False
    n1 = np.linalg.norm(w1[mask], axis=1)
    n2 = np.linalg.norm(w2[mask], axis=1)
    m1 = float(np.median(n1[n1 > 1e-8]))
    m2 = float(np.median(n2[n2 > 1e-8]))
    scale = m1 / max(m2, 1e-8)
    print(f"[Blend] median ||W_s1||={m1:.4f}, ||W_s2||={m2:.4f} -> scale={scale:.4f}, alpha={alpha:.2f}")
    out = (alpha * (w2 * scale) + (1.0 - alpha) * w1).astype(np.float32)
    if pad_id is not None:
        out[pad_id] = 0.0
    return out


def norm_matched_soup(
    mats: dict[str, np.ndarray],
    weights: dict[str, float],
    ref_key: str,
    pad_id: int | None = 0,
) -> np.ndarray:
    """
    Rescale every checkpoint matrix in `mats` to match `ref_key`'s median non-pad L2 norm,
    then compute their normalized weighted linear combination.
    """
    if ref_key not in mats:
        ref_key = next(iter(mats))
    ref = mats[ref_key]
    mask = np.ones(ref.shape[0], dtype=bool)
    if pad_id is not None:
        mask[pad_id] = False

    ref_n = np.linalg.norm(ref[mask], axis=1)
    ref_med = float(np.median(ref_n[ref_n > 1e-8]))

    active_weights = {k: v for k, v in weights.items() if v > 0.0}
    total_w = sum(active_weights.values())
    if total_w <= 0.0:
        raise ValueError("Soup recipe weights must sum to > 0.")

    out = np.zeros_like(ref, dtype=np.float32)
    for k, alpha in active_weights.items():
        if k not in mats:
            raise KeyError(f"Recipe references unknown checkpoint '{k}'. Available: {sorted(mats)}")
        w = mats[k]
        if w.shape != ref.shape:
            raise ValueError(f"Shape mismatch for '{k}': {w.shape} != {ref.shape}")
        wn = np.linalg.norm(w[mask], axis=1)
        w_med = float(np.median(wn[wn > 1e-8]))
        out += (alpha / total_w) * (w * (ref_med / max(w_med, 1e-8)))

    if pad_id is not None:
        out[pad_id] = 0.0
    return out.astype(np.float32)


def eval_nanobeir_en(
    ev: NanoBEIREvaluator,
    vecs: np.ndarray,
    tokenizer,
    device: str,
    max_length: int | None = None,
) -> dict[str, float]:
    """Evaluate an in-memory embedding matrix on NanoBEIR-en."""
    se = StaticEmbedding(tokenizer, embedding_weights=vecs)
    st = SentenceTransformer(modules=[se], device=device)
    if max_length is not None and hasattr(st, "max_seq_length"):
        st.max_seq_length = max_length
    raw = ev(st)

    def _pick(suffix: str) -> float:
        for k, v in raw.items():
            if "mean" in k.lower() and k.endswith(suffix):
                return float(v) * 100.0
        vals = [float(v) * 100.0 for k, v in raw.items() if k.endswith(suffix)]
        return float(np.mean(vals))

    return {
        "nDCG@10": _pick("ndcg@10"),
        "MRR@10": _pick("mrr@10"),
        "Recall@10": _pick("recall@10"),
    }


def run_soup(
    checkpoints: dict[str, Path],
    recipes: dict[str, dict[str, float]],
    output_dir: Path,
    ref_key: str | None = None,
    eval_nanobeir: bool = False,
    batch_size: int = 1024,
    device: str | None = None,
) -> tuple[Path, str]:
    """Build one or more norm-matched soups, optionally rank via NanoBEIR, and save the winner."""
    if not checkpoints:
        raise ValueError("At least one checkpoint must be provided.")
    if not recipes:
        raise ValueError("At least one soup recipe must be provided.")

    loaded: dict[str, np.ndarray] = {}
    max_lengths: dict[str, int | None] = {}
    tokenizer = None
    for name, ckpt_path in checkpoints.items():
        vecs, tok, max_length = load_weights(ckpt_path)
        loaded[name] = vecs
        max_lengths[name] = max_length
        if tokenizer is None:
            tokenizer = tok

    pad_id, _ = resolve_special_token_ids(tokenizer)
    resolved_ref = ref_key or next(iter(loaded))
    resolved_max_length = max_lengths.get(resolved_ref, next(iter(max_lengths.values())))

    if len(recipes) == 1 and not eval_nanobeir:
        recipe_name, w_cfg = next(iter(recipes.items()))
        vecs = norm_matched_soup(loaded, w_cfg, ref_key=resolved_ref, pad_id=pad_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        build_static_model(
            vecs, tokenizer, normalize=True, max_length=resolved_max_length
        ).save_pretrained(str(output_dir))
        print(f"[Soup] Saved '{recipe_name}' -> {output_dir}")
        return output_dir, recipe_name

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ev = NanoBEIREvaluator(
        batch_size=batch_size,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        precision_recall_at_k=[10],
        write_csv=False,
        show_progress_bar=False,
    )

    rows: dict[str, dict[str, float]] = {}
    best_name, best_score, best_vecs = "", -1.0, None
    for recipe_name, w_cfg in recipes.items():
        vecs = norm_matched_soup(loaded, w_cfg, ref_key=resolved_ref, pad_id=pad_id)
        res = eval_nanobeir_en(ev, vecs, tokenizer, resolved_device, max_length=resolved_max_length)
        rows[recipe_name] = res
        print(
            f"{recipe_name:26s} | nDCG@10={res['nDCG@10']:.2f} | "
            f"MRR@10={res['MRR@10']:.2f} | Recall@10={res['Recall@10']:.2f}"
        )
        if res["nDCG@10"] > best_score:
            best_score, best_name, best_vecs = res["nDCG@10"], recipe_name, vecs

    df = pd.DataFrame(rows).T.sort_values("nDCG@10", ascending=False)
    print("\n" + "=" * 72)
    print(df.round(2).to_string())

    assert best_vecs is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    build_static_model(
        best_vecs, tokenizer, normalize=True, max_length=resolved_max_length
    ).save_pretrained(str(output_dir))
    df.round(4).to_csv(output_dir / "soup_sweep.csv")
    print(f"\n[Soup] Saved best '{best_name}' ({best_score:.2f} nDCG@10) -> {output_dir}")
    return output_dir, best_name
