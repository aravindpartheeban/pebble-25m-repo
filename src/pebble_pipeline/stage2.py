"""Stage 2a (Contrastive Pretraining) & Stage 2b (Hard-Negative MRL InfoNCE + KL Distillation)."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model2vec import StaticModel
from tokenizers import Tokenizer
from tqdm import tqdm

from .data import (
    IndexedKDExample,
    StreamingPairSubset,
    discover_pretrain_subsets,
    load_hard_neg_dataset,
)
from .soup import norm_matched_blend
from .utils import (
    build_static_model,
    cosine_lr_lambda,
    encode_ragged,
    get_device,
    resolve_special_token_ids,
    seed_everything,
    verify_sanity,
)


@dataclass
class Stage2Config:
    stage1_checkpoint: Path
    output_dir: Path = Path("./outputs/stage2")

    # Stage 2a (optional contrastive pair pretraining)
    pretrain_pairs_dir: Path | None = None
    pretrain_q_col: str | None = None
    pretrain_d_col: str | None = None
    stage2a_batch_size: int = 2048
    stage2a_lr: float = 0.05
    stage2a_max_steps: int = 2500
    stage2a_max_subset_rows: int = 500_000

    # Stage 2b (hard-negative MRL InfoNCE + optional KL)
    ft_source: str | None = "lightonai/embeddings-fine-tuning-filtered-en"
    ft_format: str = "auto"  # auto | lighton | triplet
    ft_config: str | None = None
    stage2b_epochs: int = 2
    stage2b_batch_size: int = 1024
    stage2b_lr: float | None = None  # defaults to 0.015 after Stage 2a, 0.05 otherwise
    max_split_rows: int = 0
    max_negs_to_load: int = 10
    num_hard_negs: int = 5
    kl_weight: float = 0.0
    kl_teacher_temp: float = 0.0

    # Shared contrastive & blending
    mrl_dims: list[int] = field(default_factory=lambda: [128, 256, 512])
    max_tokens: int = 512
    token_dropout: float = 0.05
    sampling_alpha: float = 0.5
    temperature: float = 0.05
    fn_margin: float = 0.05
    blend_alpha: float = 0.8

    device: str | None = None
    seed: int = 42
    smoke_test: bool = False


def resolve_mrl_dims(requested_dims: list[int], full_dim: int) -> list[int]:
    """Keep valid Matryoshka dimensions <= full_dim and ensure full_dim is included."""
    dims = sorted({d for d in requested_dims if 0 < d <= full_dim})
    if full_dim not in dims:
        dims.append(full_dim)
    return dims


def mrl_masked_infonce_and_kl(
    q_full: torch.Tensor,
    p_full: torch.Tensor,
    n_full: torch.Tensor,
    num_negs_per_q: int,
    teacher_scores: torch.Tensor | None,
    mrl_dims: list[int],
    temperature: float,
    fn_margin: float,
    kl_weight: float,
    teacher_temp: float,
) -> tuple[torch.Tensor, float, float]:
    """
    Joint Matryoshka loss across mrl_dims:
      1) False-negative masked InfoNCE over [B positives + B*K explicit hard negatives]
      2) Optional listwise KL distillation over each query's [1 positive + K hard negatives]
    """
    bsz = q_full.size(0)
    labels = torch.arange(bsz, device=q_full.device)
    total_loss = torch.zeros((), device=q_full.device)
    last_fn = 0.0
    last_kl = 0.0

    for d in mrl_dims:
        q = F.normalize(q_full[:, :d], p=2, dim=-1, eps=1e-8)
        p = F.normalize(p_full[:, :d], p=2, dim=-1, eps=1e-8)
        n = F.normalize(n_full[:, :d], p=2, dim=-1, eps=1e-8)

        docs = torch.cat([p, n], dim=0)
        sim = q @ docs.T
        pos_sim = sim.diagonal().unsqueeze(1)

        if fn_margin >= 0.0:
            diag_mask = torch.zeros_like(sim, dtype=torch.bool)
            diag_mask[labels, labels] = True
            fn_mask = (sim > (pos_sim + fn_margin)) & (~diag_mask)
            last_fn = float(fn_mask.float().mean().item())
            sim_masked = sim.masked_fill(fn_mask, -1e4)
        else:
            sim_masked = sim

        dim_loss = F.cross_entropy(sim_masked / temperature, labels)

        if kl_weight > 0.0 and teacher_scores is not None and num_negs_per_q > 0:
            n_reshaped = n.view(bsz, num_negs_per_q, d)
            cands = torch.cat([p.unsqueeze(1), n_reshaped], dim=1)
            cand_sim = (q.unsqueeze(1) * cands).sum(dim=-1)
            log_p_student = F.log_softmax(cand_sim / temperature, dim=-1)
            p_teacher = F.softmax(teacher_scores / teacher_temp, dim=-1)
            kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean")
            dim_loss = dim_loss + kl_weight * kl
            last_kl = float(kl.item())

        total_loss = total_loss + dim_loss

    return total_loss / len(mrl_dims), last_fn, last_kl


@torch.no_grad()
def eval_hard_neg_r1(
    emb_bag: nn.EmbeddingBag,
    tokenizer: Tokenizer,
    doc_maps: dict[str, dict[int, str]],
    val_splits: dict[str, list[IndexedKDExample]],
    max_tokens: int,
    device: torch.device,
    unk_id: int,
) -> float:
    """Evaluate macro Recall@1 against each validation query's candidate pool."""
    emb_bag.eval()
    split_r1s: list[float] = []
    for split_name, examples in val_splits.items():
        dmap = doc_maps[split_name]
        hits = 0
        for ex in examples:
            cands = [dmap[ex.pos_id]] + [dmap[nid] for nid in ex.neg_ids]
            q_id, q_off = encode_ragged(tokenizer, [ex.query], max_tokens, fallback_id=unk_id)
            c_id, c_off = encode_ragged(tokenizer, cands, max_tokens, fallback_id=unk_id)
            qv = F.normalize(emb_bag(q_id.to(device), q_off.to(device)), p=2, dim=-1)
            cv = F.normalize(emb_bag(c_id.to(device), c_off.to(device)), p=2, dim=-1)
            if int(torch.argmax((qv @ cv.T).squeeze(0)).item()) == 0:
                hits += 1
        split_r1s.append(hits / max(1, len(examples)))
    emb_bag.train()
    return float(np.mean(split_r1s)) if split_r1s else 0.0


def run_stage2a_pretrain(
    emb_bag: nn.EmbeddingBag,
    tokenizer: Tokenizer,
    mrl_dims: list[int],
    cfg: Stage2Config,
    device: torch.device,
    pad_id: int | None,
    unk_id: int,
) -> Path:
    """Run streaming temperature-balanced multi-domain MRL InfoNCE pretraining."""
    assert cfg.pretrain_pairs_dir is not None
    print("\n" + "=" * 72)
    print("STAGE 2a: STREAMING MULTI-DOMAIN MRL CONTRASTIVE PRETRAINING")
    print("=" * 72)

    raw_subsets = discover_pretrain_subsets(cfg.pretrain_pairs_dir)
    if not raw_subsets:
        raise FileNotFoundError(f"No .parquet shards found under {cfg.pretrain_pairs_dir}")

    batch_size = 128 if cfg.smoke_test else cfg.stage2a_batch_size
    streams: list[StreamingPairSubset] = []
    for idx, (name, files) in enumerate(sorted(raw_subsets.items())):
        st = StreamingPairSubset(
            name,
            files,
            batch_size=batch_size,
            seed=cfg.seed + idx,
            q_col=cfg.pretrain_q_col,
            d_col=cfg.pretrain_d_col,
        )
        if st.valid:
            streams.append(st)

    if not streams:
        raise RuntimeError(
            f"Found {len(raw_subsets)} subsets under {cfg.pretrain_pairs_dir}, "
            "but none contained valid query/document columns."
        )

    total_rows = sum(s.num_rows for s in streams)
    eff_sizes = [
        min(s.num_rows, cfg.stage2a_max_subset_rows) if cfg.stage2a_max_subset_rows > 0 else s.num_rows
        for s in streams
    ]
    weights = [sz ** cfg.sampling_alpha for sz in eff_sizes]
    probs = [w / sum(weights) for w in weights]

    full_epoch_steps = max(1, total_rows // max(1, batch_size))
    if cfg.smoke_test:
        total_steps = 15
    elif cfg.stage2a_max_steps > 0:
        total_steps = cfg.stage2a_max_steps
    else:
        total_steps = full_epoch_steps

    print(
        f"[Stage 2a] Active subsets={len(streams)} | total_rows={total_rows:,} | "
        f"batch_size={batch_size} | steps={total_steps:,}"
    )

    optimizer = torch.optim.AdamW(emb_bag.parameters(), lr=cfg.stage2a_lr, weight_decay=0.0)
    warmup = max(1, int(0.05 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: cosine_lr_lambda(s, warmup, total_steps))

    rng = random.Random(cfg.seed + 101)
    emb_bag.train()
    pbar = tqdm(range(total_steps), desc="Stage 2a Pairs")
    run_loss, run_fn, log_n = 0.0, 0.0, 0

    for step in pbar:
        st = rng.choices(streams, weights=probs, k=1)[0]
        clean_q, clean_d = st.next_batch()

        q_id, q_off = encode_ragged(tokenizer, clean_q, cfg.max_tokens, cfg.token_dropout, fallback_id=unk_id)
        d_id, d_off = encode_ragged(tokenizer, clean_d, cfg.max_tokens, cfg.token_dropout, fallback_id=unk_id)

        optimizer.zero_grad(set_to_none=True)
        qf = emb_bag(q_id.to(device, non_blocking=True), q_off.to(device, non_blocking=True))
        df = emb_bag(d_id.to(device, non_blocking=True), d_off.to(device, non_blocking=True))

        bsz = qf.size(0)
        labels = torch.arange(bsz, device=device)
        diag = torch.eye(bsz, dtype=torch.bool, device=device)
        loss = torch.zeros((), device=device)
        last_fn = 0.0

        for dim in mrl_dims:
            qn = F.normalize(qf[:, :dim], p=2, dim=-1, eps=1e-8)
            dn = F.normalize(df[:, :dim], p=2, dim=-1, eps=1e-8)
            sim = qn @ dn.T
            if cfg.fn_margin >= 0.0:
                fn_mask = (sim > (sim.diagonal().unsqueeze(1) + cfg.fn_margin)) & (~diag)
                last_fn = float(fn_mask.float().mean().item())
                sim = sim.masked_fill(fn_mask, -1e4)
            loss = loss + F.cross_entropy(sim / cfg.temperature, labels)

        loss = loss / len(mrl_dims)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite Stage 2a loss at step {step} ({st.name})")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(emb_bag.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if pad_id is not None:
            with torch.no_grad():
                emb_bag.weight[pad_id].zero_()

        run_loss += float(loss.item())
        run_fn += last_fn
        log_n += 1
        if (step + 1) % (5 if cfg.smoke_test else 50) == 0:
            pbar.set_postfix(
                loss=f"{run_loss / log_n:.3f}",
                fn=f"{100.0 * run_fn / log_n:.2f}%",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
            run_loss, run_fn, log_n = 0.0, 0.0, 0

    w_s2a = emb_bag.weight.detach().float().cpu().numpy().astype(np.float32)
    if pad_id is not None:
        w_s2a[pad_id] = 0.0
    s2a_out = cfg.output_dir / "stage2a_pretrain"
    s2a_out.mkdir(parents=True, exist_ok=True)
    build_static_model(
        w_s2a, tokenizer, normalize=True, max_length=cfg.max_tokens
    ).save_pretrained(str(s2a_out))
    print(f"[Stage 2a] Saved checkpoint -> {s2a_out}")
    return s2a_out


def run_stage2b_hard_neg(
    emb_bag: nn.EmbeddingBag,
    tokenizer: Tokenizer,
    mrl_dims: list[int],
    cfg: Stage2Config,
    device: torch.device,
    pad_id: int | None,
    unk_id: int,
    ran_stage2a: bool,
) -> tuple[np.ndarray, Path, float, float]:
    """Run hard-negative MRL InfoNCE (+ optional cross-encoder KL distillation)."""
    assert cfg.ft_source is not None
    print("\n" + "=" * 72)
    print(
        "STAGE 2b: HARD-NEGATIVE MRL INFONCE"
        + (" + KL DISTILLATION" if cfg.kl_weight > 0 else " (PURE CONTRASTIVE)")
    )
    print("=" * 72)

    doc_maps, train_splits, val_splits, auto_t = load_hard_neg_dataset(
        source=cfg.ft_source,
        dataset_format=cfg.ft_format,
        dataset_config=cfg.ft_config,
        max_split_rows=cfg.max_split_rows,
        max_negs=cfg.max_negs_to_load,
        max_chars=2500,
        val_per_split=16 if cfg.smoke_test else 64,
        smoke_test=cfg.smoke_test,
        seed=cfg.seed,
    )
    teacher_temp = cfg.kl_teacher_temp if cfg.kl_teacher_temp > 0 else auto_t

    if cfg.kl_weight > 0.0:
        if cfg.num_hard_negs <= 0:
            raise ValueError(
                "kl_weight > 0 requires num_hard_negs > 0: the KL term distills over "
                "[positive, *hard_negatives], so at least one hard negative per query is needed."
            )
        scored = sum(
            1
            for split in train_splits.values()
            for ex in split
            if ex.rerank_scores and len(ex.rerank_scores) == 1 + len(ex.neg_ids)
        )
        total_train = sum(len(split) for split in train_splits.values())
        if scored == 0:
            raise ValueError(
                f"kl_weight={cfg.kl_weight} > 0 but none of the {total_train:,} training examples have "
                "aligned `rerank_scores` (expected 1 + len(negatives) values). "
                "Set kl_weight=0 for pure InfoNCE, or provide a dataset with teacher scores."
            )
        print(
            f"[Stage 2b] KL teacher scores available on {scored:,}/{total_train:,} train examples "
            f"({100.0 * scored / max(1, total_train):.1f}%)"
        )

    base_r1 = eval_hard_neg_r1(emb_bag, tokenizer, doc_maps, val_splits, cfg.max_tokens, device, unk_id)
    print(f"[Stage 2b] Pre-2b held-out hard-neg R@1 (macro across {len(val_splits)} splits): {base_r1:.4f}")

    epochs = 1 if cfg.smoke_test else cfg.stage2b_epochs
    batch_size = min(64, max(4, sum(len(v) for v in train_splits.values()))) if cfg.smoke_test else cfg.stage2b_batch_size
    names = sorted(train_splits.keys())
    sizes = [len(train_splits[n]) for n in names]
    steps_per_epoch = max(1, sum(sizes) // max(1, batch_size))
    max_steps = 20 if cfg.smoke_test else None
    total_steps = max_steps if max_steps is not None else steps_per_epoch * epochs

    lr = cfg.stage2b_lr if cfg.stage2b_lr is not None else (0.015 if ran_stage2a else 0.05)
    print(
        f"[Stage 2b] batch_size={batch_size} | lr={lr} | epochs={epochs} | "
        f"steps/epoch={steps_per_epoch:,} | kl_weight={cfg.kl_weight}"
    )

    optimizer = torch.optim.AdamW(emb_bag.parameters(), lr=lr, weight_decay=0.0)
    warmup = max(1, int(0.10 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: cosine_lr_lambda(s, warmup, total_steps))

    weights = [s ** cfg.sampling_alpha for s in sizes]
    probs = [w / sum(weights) for w in weights]
    k_negs = cfg.num_hard_negs

    global_step = 0
    kl_steps = 0
    val_r1 = base_r1
    rng = random.Random(cfg.seed + 77)
    emb_bag.train()

    for epoch in range(epochs):
        pools = {n: list(range(len(train_splits[n]))) for n in names}
        for p_idx in pools.values():
            rng.shuffle(p_idx)
        cursors = {n: 0 for n in names}

        n_steps_this_ep = max_steps if max_steps is not None else steps_per_epoch
        pbar = tqdm(range(n_steps_this_ep), desc=f"Stage 2b Epoch {epoch + 1}/{epochs}")
        run_loss, run_kl, run_fn, log_n = 0.0, 0.0, 0.0, 0

        for _ in pbar:
            split_name = rng.choices(names, weights=probs, k=1)[0]
            data = train_splits[split_name]
            dmap = doc_maps[split_name]
            pool = pools[split_name]
            cur = cursors[split_name]

            batch: list[IndexedKDExample] = []
            seen_q: set[str] = set()
            seen_p: set[int] = set()
            attempts = 0
            while len(batch) < batch_size and attempts < min(len(data), batch_size * 2):
                if cur >= len(pool):
                    rng.shuffle(pool)
                    cur = 0
                ex = data[pool[cur]]
                cur += 1
                attempts += 1
                if ex.query in seen_q or ex.pos_id in seen_p or len(ex.neg_ids) < k_negs:
                    continue
                seen_q.add(ex.query)
                seen_p.add(ex.pos_id)
                batch.append(ex)
            cursors[split_name] = cur
            if len(batch) < 2:
                continue

            queries = [ex.query for ex in batch]
            positives = [dmap[ex.pos_id] for ex in batch]
            neg_texts: list[str] = []
            t_scores_rows: list[list[float]] = []
            all_have_scores = cfg.kl_weight > 0.0

            for ex in batch:
                chosen_idx = rng.sample(range(len(ex.neg_ids)), k=k_negs)
                for ci in chosen_idx:
                    neg_texts.append(dmap[ex.neg_ids[ci]])
                if all_have_scores and ex.rerank_scores and len(ex.rerank_scores) == 1 + len(ex.neg_ids):
                    row_s = [ex.rerank_scores[0]] + [ex.rerank_scores[1 + ci] for ci in chosen_idx]
                    t_scores_rows.append(row_s)
                else:
                    all_have_scores = False

            q_id, q_off = encode_ragged(tokenizer, queries, cfg.max_tokens, cfg.token_dropout, fallback_id=unk_id)
            p_id, p_off = encode_ragged(tokenizer, positives, cfg.max_tokens, cfg.token_dropout, fallback_id=unk_id)
            n_id, n_off = encode_ragged(tokenizer, neg_texts, cfg.max_tokens, cfg.token_dropout, fallback_id=unk_id)
            t_scores = torch.tensor(t_scores_rows, dtype=torch.float32, device=device) if all_have_scores else None
            if t_scores is not None:
                kl_steps += 1

            optimizer.zero_grad(set_to_none=True)
            qf = emb_bag(q_id.to(device, non_blocking=True), q_off.to(device, non_blocking=True))
            pf = emb_bag(p_id.to(device, non_blocking=True), p_off.to(device, non_blocking=True))
            nf = emb_bag(n_id.to(device, non_blocking=True), n_off.to(device, non_blocking=True))

            loss, fn_frac, kl_val = mrl_masked_infonce_and_kl(
                q_full=qf,
                p_full=pf,
                n_full=nf,
                num_negs_per_q=k_negs,
                teacher_scores=t_scores,
                mrl_dims=mrl_dims,
                temperature=cfg.temperature,
                fn_margin=cfg.fn_margin,
                kl_weight=cfg.kl_weight,
                teacher_temp=teacher_temp,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Stage 2b loss at step {global_step}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(emb_bag.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if pad_id is not None:
                with torch.no_grad():
                    emb_bag.weight[pad_id].zero_()

            global_step += 1
            run_loss += float(loss.item())
            run_kl += kl_val
            run_fn += fn_frac
            log_n += 1
            if global_step % (5 if cfg.smoke_test else 50) == 0:
                pbar.set_postfix(
                    loss=f"{run_loss / log_n:.3f}",
                    kl=f"{run_kl / log_n:.3f}",
                    fn=f"{100.0 * run_fn / log_n:.2f}%",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )
                run_loss, run_kl, run_fn, log_n = 0.0, 0.0, 0.0, 0

        val_r1 = eval_hard_neg_r1(emb_bag, tokenizer, doc_maps, val_splits, cfg.max_tokens, device, unk_id)
        print(f"[Stage 2b] Epoch {epoch + 1} | held-out hard-neg R@1={val_r1:.4f} (pre-2b={base_r1:.4f})")

    if cfg.kl_weight > 0.0:
        coverage = 100.0 * kl_steps / max(1, global_step)
        print(f"[Stage 2b] KL distillation applied on {kl_steps:,}/{global_step:,} steps ({coverage:.1f}%)")
        if kl_steps == 0:
            raise RuntimeError(
                "kl_weight > 0 but KL distillation was applied on 0 steps: no batch had aligned "
                "teacher scores. Aborting instead of silently saving an un-distilled checkpoint."
            )
        if coverage < 99.9:
            print(
                "[Stage 2b] WARNING: some batches lacked aligned `rerank_scores`, so KL was skipped for "
                "those steps. Filter the dataset or fix the score alignment for full KL coverage."
            )

    w_s2 = emb_bag.weight.detach().float().cpu().numpy().astype(np.float32)
    if pad_id is not None:
        w_s2[pad_id] = 0.0
    s2_out = cfg.output_dir / "stage2_contrastive"
    s2_out.mkdir(parents=True, exist_ok=True)
    build_static_model(
        w_s2, tokenizer, normalize=True, max_length=cfg.max_tokens
    ).save_pretrained(str(s2_out))
    print(f"[Stage 2b] Saved -> {s2_out}")
    return w_s2, s2_out, base_r1, val_r1


def run_stage2(cfg: Stage2Config) -> dict[str, Path]:
    """Execute Stage 2a (optional), Stage 2b (optional), and optional 2-way norm-matched blend."""
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    with (cfg.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()}, f, indent=2)

    print(f"[Stage 2] Loading Stage-1 checkpoint from {cfg.stage1_checkpoint}...")
    s1_model = StaticModel.from_pretrained(str(cfg.stage1_checkpoint))
    w_s1 = np.asarray(s1_model.embedding, dtype=np.float32).copy()
    tokenizer: Tokenizer = s1_model.tokenizer
    s1_max_length = getattr(s1_model, "max_length", None)
    vocab_size, dim = w_s1.shape
    pad_id, unk_id = resolve_special_token_ids(tokenizer)
    print(f"[Stage 2] Loaded shape=({vocab_size:,}, {dim}) | pad_id={pad_id} | unk_id={unk_id} | device={device}")

    s1_copy_dir = cfg.output_dir / "stage1_tokenlearn"
    if s1_copy_dir.resolve() != cfg.stage1_checkpoint.resolve():
        s1_copy_dir.mkdir(parents=True, exist_ok=True)
        build_static_model(
            w_s1, tokenizer, normalize=True, max_length=s1_max_length
        ).save_pretrained(str(s1_copy_dir))

    mrl_dims = resolve_mrl_dims(cfg.mrl_dims, dim)
    print(f"[Stage 2] Matryoshka dims: {mrl_dims}")

    emb_bag = nn.EmbeddingBag(vocab_size, dim, mode="mean", sparse=False).to(device)
    with torch.no_grad():
        emb_bag.weight.copy_(torch.from_numpy(w_s1))

    ckpts: dict[str, Path] = {"stage1": s1_copy_dir}
    ran_stage2a = False
    if cfg.pretrain_pairs_dir is not None:
        ckpts["stage2a"] = run_stage2a_pretrain(emb_bag, tokenizer, mrl_dims, cfg, device, pad_id, unk_id)
        ran_stage2a = True

    base_r1, final_r1 = 0.0, 0.0
    w_last: np.ndarray | None = None
    if cfg.ft_source:
        w_last, s2b_dir, base_r1, final_r1 = run_stage2b_hard_neg(
            emb_bag, tokenizer, mrl_dims, cfg, device, pad_id, unk_id, ran_stage2a
        )
        ckpts["stage2b"] = s2b_dir
    elif ran_stage2a:
        w_last = emb_bag.weight.detach().float().cpu().numpy().astype(np.float32)

    if w_last is not None and 0.0 < cfg.blend_alpha <= 1.0:
        w_final = norm_matched_blend(w_s1, w_last, alpha=cfg.blend_alpha, pad_id=pad_id)
        final_out = cfg.output_dir / "final_blended"
        final_out.mkdir(parents=True, exist_ok=True)
        build_static_model(
            w_final, tokenizer, normalize=True, max_length=cfg.max_tokens
        ).save_pretrained(str(final_out))
        print(f"[Blend] Saved -> {final_out}")
        ckpts["final_blended"] = final_out

    verify_sanity(ckpts)

    tsv_path = cfg.output_dir.parent / "results.tsv"
    write_hdr = not tsv_path.exists()
    with tsv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        if write_hdr:
            w.writerow(
                [
                    "run_id",
                    "hypothesis",
                    "method",
                    "dataset",
                    "lr",
                    "epochs_or_steps",
                    "baseline_metric",
                    "eval_metric",
                    "status",
                    "notes",
                ]
            )
        w.writerow(
            [
                cfg.output_dir.name,
                f"{'s2a+' if ran_stage2a else ''}s2b_kl={cfg.kl_weight}+blend={cfg.blend_alpha:.2f}",
                f"static_{vocab_size}x{dim}_mrl",
                f"{'pretrain+' if ran_stage2a else ''}{cfg.ft_source or 'none'}",
                f"s2a={cfg.stage2a_lr},s2b={cfg.stage2b_lr or ('0.015' if ran_stage2a else '0.05')}",
                f"s2a={cfg.stage2a_max_steps}st,s2b={cfg.stage2b_epochs}ep" if not cfg.smoke_test else "smoke",
                f"pre_2b_r1={base_r1:.4f}",
                f"post_2b_r1={final_r1:.4f}",
                "smoke_pass" if cfg.smoke_test else "completed",
                f"stage1_src={cfg.stage1_checkpoint.name}",
            ]
        )
    return ckpts
