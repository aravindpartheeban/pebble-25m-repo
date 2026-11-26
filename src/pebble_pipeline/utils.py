"""Shared utilities for static embedding distillation, training, and blending."""

from __future__ import annotations

import inspect
import math
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from model2vec import StaticModel
from tokenizers import Tokenizer


def seed_everything(seed: int) -> None:
    """Set random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str | None = None) -> torch.device:
    """Resolve the compute device, defaulting to CUDA when available."""
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_special_token_ids(tokenizer: Tokenizer) -> tuple[int | None, int]:
    """Return (pad_id, unk_id) from a Hugging Face tokenizers.Tokenizer."""
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        pad_id = tokenizer.token_to_id("<pad>")
    unk_id = tokenizer.token_to_id("[UNK]")
    if unk_id is None:
        unk_id = tokenizer.token_to_id("<unk>")
    if unk_id is None:
        unk_id = 1 if pad_id == 0 else 0
    return pad_id, unk_id


_UNSET = object()


def build_static_model(
    vectors: np.ndarray,
    tokenizer: Tokenizer,
    normalize: bool = True,
    max_length: int | None | object = _UNSET,
) -> StaticModel:
    """Construct a model2vec StaticModel compatible across model2vec versions.

    ``max_length`` is persisted into the saved config so inference truncation matches the
    training ``max_tokens``. Omit it to keep model2vec's default; pass ``None`` to disable
    truncation explicitly.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    params = inspect.signature(StaticModel.__init__).parameters
    kwargs: dict = {"tokenizer": tokenizer}
    kwargs["vectors" if "vectors" in params else "embedding"] = vectors
    if "normalize" in params:
        kwargs["normalize"] = normalize
    if max_length is not _UNSET and "max_length" in params:
        kwargs["max_length"] = max_length
    return StaticModel(**kwargs)


def arrow_embeddings_to_numpy(col: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """Convert a PyArrow FixedSizeList or List embedding column to float32 ndarray."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    if pa.types.is_fixed_size_list(col.type):
        dim = col.type.list_size
        flat = col.values.to_numpy(zero_copy_only=False)
        return np.asarray(flat, dtype=np.float32).reshape(len(col), dim)
    if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
        offsets = col.offsets.to_numpy(zero_copy_only=False)
        if len(offsets) > 1:
            diffs = np.diff(offsets)
            if np.all(diffs == diffs[0]) and diffs[0] > 0:
                flat = col.values.to_numpy(zero_copy_only=False)
                return np.asarray(flat, dtype=np.float32).reshape(len(col), int(diffs[0]))
    return np.asarray(col.to_pylist(), dtype=np.float32)


def encode_ragged(
    tokenizer: Tokenizer,
    texts: list[str],
    max_tokens: int,
    token_dropout: float = 0.0,
    fallback_id: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize texts into flat token IDs and offsets for nn.EmbeddingBag."""
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
    flat_ids: list[int] = []
    offsets: list[int] = []
    for enc in encodings:
        ids = enc.ids[:max_tokens]
        if token_dropout > 0.0 and len(ids) > 1:
            kept = [tid for tid in ids if random.random() >= token_dropout]
            ids = kept if kept else [random.choice(ids)]
        if not ids:
            ids = [fallback_id]
        offsets.append(len(flat_ids))
        flat_ids.extend(ids)
    return torch.tensor(flat_ids, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


def cosine_lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup followed by cosine decay to zero."""
    if total_steps <= 1:
        return 1.0
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def verify_sanity(ckpts: dict[str, Path]) -> None:
    """Reload saved StaticModel checkpoints and print pairwise semantic sanity checks."""
    print("\n" + "=" * 72)
    print("POST-SAVE RELOAD & SEMANTIC SANITY CHECK")
    print("=" * 72)
    pairs = [
        (
            "Paraphrase",
            "How do neural networks update weights during backpropagation?",
            "Gradient descent adjusts layer parameters using chain-rule derivatives.",
        ),
        (
            "Technical QA",
            "Why does a Kubernetes pod enter CrashLoopBackOff?",
            "The container keeps exiting with an error and kubelet repeatedly restarts it.",
        ),
        (
            "Finance QA",
            "What causes bond prices to drop when interest rates rise?",
            "New bonds offer higher yields, making older lower-coupon bonds less attractive.",
        ),
        (
            "Unrelated",
            "How do I configure SSL certificates in Nginx?",
            "Sourdough fermentation relies on wild yeast and lactic acid bacteria.",
        ),
    ]
    for label, path in ckpts.items():
        model = StaticModel.from_pretrained(str(path))
        vecs = np.asarray(model.embedding)
        if not np.isfinite(vecs).all():
            raise RuntimeError(f"Non-finite weights detected in {path}")
        print(f"\n[{label}] {path} | shape={vecs.shape} | params={vecs.size:,}")
        for pname, a, b in pairs:
            emb = model.encode([a, b])
            na = emb[0] / (np.linalg.norm(emb[0]) + 1e-12)
            nb = emb[1] / (np.linalg.norm(emb[1]) + 1e-12)
            print(f"  {pname:14s}: cos={float(np.dot(na, nb)):+.4f}")
