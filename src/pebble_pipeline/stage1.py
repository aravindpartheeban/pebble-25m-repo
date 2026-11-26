"""Stage 0 (Tokenizer) & Stage 1 (Tokenlearn Distillation + PCA + SIF)."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tqdm import tqdm

from .utils import (
    arrow_embeddings_to_numpy,
    build_static_model,
    cosine_lr_lambda,
    encode_ragged,
    get_device,
    resolve_special_token_ids,
    seed_everything,
)


@dataclass
class Stage1Config:
    stage1_dir: Path
    output_dir: Path = Path("./outputs/run")
    text_col: str = "text"
    embedding_col: str = "embedding"

    # Stage 0: Tokenizer
    tokenizer_path: Path | None = None
    vocab_size: int = 48_000
    min_frequency: int = 3
    tokenizer_texts: int = 500_000
    lowercase: bool = True
    strip_accents: bool = True

    # Stage 1: Tokenlearn
    dim: int = 512
    pca_fit_samples: int = 100_000
    epochs: int = 5
    batch_size: int = 512
    lr: float = 3e-3
    max_tokens: int = 512
    token_dropout: float = 0.05
    val_size: int = 4096
    apply_sif: bool = True
    sif_a: float = 1e-3

    device: str | None = None
    seed: int = 42
    smoke_test: bool = False


def train_or_load_tokenizer(
    stage1_shards: list[Path],
    output_dir: Path,
    cfg: Stage1Config,
) -> Tokenizer:
    """Load an explicit/cached tokenizer or train a WordPiece tokenizer from Parquet texts."""
    if cfg.tokenizer_path is not None:
        print(f"[Stage 0] Loaded user tokenizer -> {cfg.tokenizer_path}")
        return Tokenizer.from_file(str(cfg.tokenizer_path))

    tok_path = output_dir / f"wordpiece_{cfg.vocab_size}.json"
    if tok_path.exists():
        print(f"[Stage 0] Loaded cached tokenizer -> {tok_path}")
        return Tokenizer.from_file(str(tok_path))

    max_texts = 10_000 if cfg.smoke_test else cfg.tokenizer_texts
    print(f"[Stage 0] Training {cfg.vocab_size:,}-vocab WordPiece tokenizer on up to {max_texts:,} texts...")
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=cfg.strip_accents,
        lowercase=cfg.lowercase,
    )
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece(prefix="##")

    def _text_iter() -> Iterator[list[str]]:
        seen = 0
        for fp in stage1_shards:
            pf = pq.ParquetFile(fp)
            for batch in pf.iter_batches(batch_size=8192, columns=[cfg.text_col]):
                texts = [t for t in batch.column(cfg.text_col).to_pylist() if t and str(t).strip()]
                if not texts:
                    continue
                if seen + len(texts) > max_texts:
                    texts = texts[: max_texts - seen]
                yield texts
                seen += len(texts)
                if seen >= max_texts:
                    return

    trainer = trainers.WordPieceTrainer(
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        continuing_subword_prefix="##",
    )
    t0 = time.time()
    tok.train_from_iterator(_text_iter(), trainer=trainer)
    output_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(tok_path))
    print(f"[Stage 0] Trained vocab={tok.get_vocab_size():,} in {time.time() - t0:.1f}s -> {tok_path}")
    return tok


class TokenlearnStudent(nn.Module):
    """Static EmbeddingBag with a foldable linear projection head for Stage-1 distillation."""

    def __init__(self, vocab_size: int, dim: int, pad_id: int | None = 0) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.EmbeddingBag(vocab_size, dim, mode="mean", sparse=False)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.pad_id is not None:
            with torch.no_grad():
                self.embedding.weight[self.pad_id].zero_()
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, flat_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embedding(flat_ids, offsets))

    @torch.no_grad()
    def export_folded_vectors(self) -> np.ndarray:
        """Fold the linear projection into the token embedding matrix."""
        w = self.embedding.weight.detach().float() @ self.proj.weight.detach().float().T
        if self.pad_id is not None:
            w[self.pad_id].zero_()
        return w.cpu().numpy().astype(np.float32)


def fit_target_pca(
    shards: list[Path],
    embedding_col: str,
    target_dim: int,
    fit_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit PCA to project teacher embeddings to target_dim (returns None if dims already match)."""
    collected: list[np.ndarray] = []
    total = 0
    for fp in shards:
        for batch in pq.ParquetFile(fp).iter_batches(batch_size=8192, columns=[embedding_col]):
            arr = arrow_embeddings_to_numpy(batch.column(embedding_col))
            arr = arr[np.isfinite(arr).all(axis=1)]
            collected.append(arr)
            total += len(arr)
            if total >= fit_samples:
                break
        if total >= fit_samples:
            break

    mat = np.concatenate(collected, axis=0)[:fit_samples]
    src_dim = mat.shape[1]
    if src_dim == target_dim:
        print(f"[Stage 1] Teacher dim ({src_dim}) == student dim ({target_dim}); skipping PCA.")
        return None
    if src_dim < target_dim:
        raise ValueError(f"Teacher dim ({src_dim}) < requested student dim ({target_dim}).")

    pca = PCA(n_components=target_dim, random_state=seed)
    pca.fit(mat)
    ev = float(pca.explained_variance_ratio_.sum()) * 100.0
    print(f"[Stage 1] Target PCA ({src_dim}d -> {target_dim}d) on {len(mat):,} rows | explained variance={ev:.2f}%")
    return pca.mean_.astype(np.float32), pca.components_.astype(np.float32)


def _project_targets(
    embs: torch.Tensor,
    pca_mean_t: torch.Tensor | None,
    pca_comp_t: torch.Tensor | None,
) -> torch.Tensor:
    if pca_mean_t is None or pca_comp_t is None:
        return embs
    return (embs - pca_mean_t) @ pca_comp_t.T


@torch.no_grad()
def eval_stage1(
    model: TokenlearnStudent,
    tokenizer: Tokenizer,
    val_texts: list[str],
    val_embs: np.ndarray,
    pca_mean_t: torch.Tensor | None,
    pca_comp_t: torch.Tensor | None,
    max_tokens: int,
    batch_size: int,
    device: torch.device,
    unk_id: int,
) -> float:
    model.eval()
    cos_sum = 0.0
    n = 0
    for i in range(0, len(val_texts), batch_size):
        bt = val_texts[i : i + batch_size]
        be = torch.from_numpy(val_embs[i : i + batch_size]).to(device)
        targets = _project_targets(be, pca_mean_t, pca_comp_t)
        ids, off = encode_ragged(tokenizer, bt, max_tokens=max_tokens, fallback_id=unk_id)
        preds = model(ids.to(device), off.to(device))
        cos_sum += float(F.cosine_similarity(preds, targets, dim=-1, eps=1e-8).sum().item())
        n += len(bt)
    model.train()
    return cos_sum / max(1, n)


def apply_sif(
    vectors: np.ndarray,
    token_counts: np.ndarray,
    sif_a: float = 1e-3,
    pad_id: int | None = 0,
) -> np.ndarray:
    """Apply median-normalized Smooth Inverse Frequency (SIF) weighting."""
    total = float(token_counts.sum())
    if total <= 0:
        return vectors
    probs = token_counts.astype(np.float64) / total
    w = sif_a / (sif_a + probs)
    active = token_counts > 0
    med = float(np.median(w[active])) if np.any(active) else 1.0
    w = (w / max(med, 1e-12)).astype(np.float32)
    out = (vectors * w[:, None]).astype(np.float32)
    if pad_id is not None:
        out[pad_id] = 0.0
    return out


def run_stage1(cfg: Stage1Config) -> tuple[np.ndarray, Tokenizer, Path, float]:
    """Run Stage 0 (tokenizer) and Stage 1 (Tokenlearn + SIF), returning (vectors, tokenizer, ckpt_dir, val_cos)."""
    print("\n" + "=" * 72)
    print("STAGE 0 & 1: TOKENIZER + TOKENLEARN DISTILLATION")
    print("=" * 72)

    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(cfg.stage1_dir.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No .parquet files found under {cfg.stage1_dir}")

    tokenizer = train_or_load_tokenizer(shards, cfg.output_dir, cfg)
    pad_id, unk_id = resolve_special_token_ids(tokenizer)
    vocab_size = tokenizer.get_vocab_size()

    if cfg.smoke_test:
        shards = shards[:1]

    pf_last = pq.ParquetFile(shards[-1])
    val_size = min(256 if cfg.smoke_test else cfg.val_size, max(2, pf_last.metadata.num_rows // 5))
    tbl = pf_last.read(columns=[cfg.text_col, cfg.embedding_col])
    sub = tbl.slice(tbl.num_rows - val_size, val_size)
    raw_vt = [t or "" for t in sub.column(cfg.text_col).to_pylist()]
    raw_ve = arrow_embeddings_to_numpy(sub.column(cfg.embedding_col))
    keep_v = [i for i, t in enumerate(raw_vt) if str(t).strip() and np.isfinite(raw_ve[i]).all()]
    val_texts = [raw_vt[i] for i in keep_v]
    val_embs = raw_ve[keep_v]

    fit_n = min(4000, pf_last.metadata.num_rows) if cfg.smoke_test else cfg.pca_fit_samples
    pca_res = fit_target_pca(shards, cfg.embedding_col, target_dim=cfg.dim, fit_samples=fit_n, seed=cfg.seed)
    pca_mean_t = torch.from_numpy(pca_res[0]).to(device) if pca_res is not None else None
    pca_comp_t = torch.from_numpy(pca_res[1]).to(device) if pca_res is not None else None

    model = TokenlearnStudent(vocab_size=vocab_size, dim=cfg.dim, pad_id=pad_id).to(device)
    token_counts = np.zeros(vocab_size, dtype=np.int64)

    total_rows = max(1, sum(pq.ParquetFile(f).metadata.num_rows for f in shards) - val_size)
    epochs = 1 if cfg.smoke_test else cfg.epochs
    batch_size = min(cfg.batch_size, max(8, total_rows // 2)) if cfg.smoke_test else cfg.batch_size
    max_steps = 25 if cfg.smoke_test else None
    est_steps = max_steps if max_steps is not None else max(1, (total_rows // max(1, batch_size)) * epochs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    warmup = max(1, int(0.05 * est_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: cosine_lr_lambda(s, warmup, est_steps))

    global_step = 0
    val_cos = 0.0
    model.train()

    for epoch in range(epochs):
        order = list(shards)
        random.Random(cfg.seed + epoch).shuffle(order)
        pbar = tqdm(
            total=max_steps if max_steps is not None else max(1, total_rows // max(1, batch_size)),
            desc=f"Stage 1 Epoch {epoch + 1}/{epochs}",
        )
        run_loss = 0.0
        log_n = 0

        for sp in order:
            pf = pq.ParquetFile(sp)
            limit = pf.metadata.num_rows - val_size if sp == shards[-1] else pf.metadata.num_rows
            seen = 0
            for batch in pf.iter_batches(batch_size=batch_size, columns=[cfg.text_col, cfg.embedding_col]):
                if seen >= limit:
                    break
                take = min(batch.num_rows, limit - seen)
                seen += batch.num_rows
                texts = batch.column(cfg.text_col).to_pylist()[:take]
                embs = arrow_embeddings_to_numpy(batch.column(cfg.embedding_col))[:take]
                keep = [i for i, t in enumerate(texts) if t and str(t).strip() and np.isfinite(embs[i]).all()]
                if len(keep) < 2:
                    continue

                bt = [texts[i] for i in keep]
                be = torch.from_numpy(embs[keep]).to(device, non_blocking=True)
                targets = _project_targets(be, pca_mean_t, pca_comp_t)

                ids, off = encode_ragged(tokenizer, bt, cfg.max_tokens, cfg.token_dropout, fallback_id=unk_id)
                if epoch == 0:
                    np.add.at(token_counts, ids.numpy(), 1)

                optimizer.zero_grad(set_to_none=True)
                preds = model(ids.to(device, non_blocking=True), off.to(device, non_blocking=True))
                loss = (1.0 - F.cosine_similarity(preds, targets, dim=-1, eps=1e-8)).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite Stage 1 loss at step {global_step}")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                if pad_id is not None:
                    with torch.no_grad():
                        model.embedding.weight[pad_id].zero_()

                global_step += 1
                run_loss += float(loss.item())
                log_n += 1
                pbar.update(1)
                if global_step % (5 if cfg.smoke_test else 100) == 0:
                    pbar.set_postfix(loss=f"{run_loss / max(1, log_n):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                    run_loss = 0.0
                    log_n = 0
                if max_steps is not None and global_step >= max_steps:
                    break
            if max_steps is not None and global_step >= max_steps:
                break
        pbar.close()

        if val_texts:
            val_cos = eval_stage1(
                model,
                tokenizer,
                val_texts,
                val_embs,
                pca_mean_t,
                pca_comp_t,
                cfg.max_tokens,
                batch_size,
                device,
                unk_id,
            )
            print(f"[Stage 1] Epoch {epoch + 1} | val_cosine_sim={val_cos:.4f}")

    folded = model.export_folded_vectors()
    if cfg.apply_sif and not cfg.smoke_test:
        folded = apply_sif(folded, token_counts, sif_a=cfg.sif_a, pad_id=pad_id)

    s1_out = cfg.output_dir / "stage1_tokenlearn"
    s1_out.mkdir(parents=True, exist_ok=True)
    build_static_model(folded, tokenizer, normalize=True, max_length=cfg.max_tokens).save_pretrained(str(s1_out))
    print(f"[Stage 1] Saved -> {s1_out} (params={folded.size:,})")
    return folded, tokenizer, s1_out, val_cos
