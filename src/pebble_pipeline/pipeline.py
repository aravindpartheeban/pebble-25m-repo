"""End-to-end YAML/JSON pipeline orchestrator for Stages 0 -> 1 -> 2 (multi-branch) -> 3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .featurize import FeaturizeConfig, run_featurize
from .soup import run_soup
from .stage1 import Stage1Config, run_stage1
from .stage2 import Stage2Config, run_stage2


def _as_path(val: str | Path | None) -> Path | None:
    return Path(val) if val is not None else None


def run_from_config(config_path: Path, smoke_test: bool = False) -> Path:
    """Execute the configured distillation, contrastive, and souping stages."""
    with config_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    seed = int(cfg.get("seed", 42))
    device = cfg.get("device")
    work_dir = Path(cfg.get("output_dir", "./outputs/pebble_run"))
    work_dir.mkdir(parents=True, exist_ok=True)

    # Optional Stage 0a: Teacher featurization
    feat_cfg_raw = cfg.get("featurize")
    features_dir: Path | None = _as_path(cfg.get("stage1", {}).get("stage1_dir"))
    if feat_cfg_raw and feat_cfg_raw.get("enabled", False):
        f_out = Path(feat_cfg_raw.get("output_dir", work_dir / "teacher_features"))
        f_cfg = FeaturizeConfig(
            model_id=feat_cfg_raw.get("model_id", "MongoDB/voyage-4-nano"),
            trust_remote_code=bool(feat_cfg_raw.get("trust_remote_code", True)),
            dim=feat_cfg_raw.get("dim", 1024),
            max_seq_length=int(feat_cfg_raw.get("max_seq_length", 512)),
            encode_mode=feat_cfg_raw.get("encode_mode", "auto"),
            dataset=feat_cfg_raw.get("dataset", "ccde/fineweb-edu-sample-500K"),
            dataset_config=feat_cfg_raw.get("dataset_config"),
            dataset_split=feat_cfg_raw.get("dataset_split", "train"),
            text_column=feat_cfg_raw.get("text_column", "text"),
            local_files=feat_cfg_raw.get("local_files"),
            output_dir=f_out,
            target_rows=int(feat_cfg_raw.get("target_rows", 500_000)),
            rows_per_file=int(feat_cfg_raw.get("rows_per_file", 100_000)),
            batch_size=int(feat_cfg_raw.get("batch_size", 64)),
            min_words=int(feat_cfg_raw.get("min_words", 25)),
            max_words=int(feat_cfg_raw.get("max_words", 320)),
            device=device,
            smoke_test=smoke_test,
        )
        run_featurize(f_cfg)
        features_dir = f_out

    # Stage 0 & 1: Tokenizer + Tokenlearn
    s1_raw = cfg.get("stage1", {})
    if features_dir is None:
        raise ValueError("Config must specify either featurize.enabled=true or stage1.stage1_dir.")

    s1_out_base = Path(s1_raw.get("output_dir", work_dir / "stage1"))
    s1_cfg = Stage1Config(
        stage1_dir=features_dir,
        output_dir=s1_out_base,
        text_col=s1_raw.get("text_col", "text"),
        embedding_col=s1_raw.get("embedding_col", "embedding"),
        tokenizer_path=_as_path(s1_raw.get("tokenizer_path")),
        vocab_size=int(s1_raw.get("vocab_size", 48_000)),
        min_frequency=int(s1_raw.get("min_frequency", 3)),
        tokenizer_texts=int(s1_raw.get("tokenizer_texts", 500_000)),
        lowercase=bool(s1_raw.get("lowercase", True)),
        strip_accents=bool(s1_raw.get("strip_accents", True)),
        dim=int(s1_raw.get("dim", 512)),
        pca_fit_samples=int(s1_raw.get("pca_fit_samples", 100_000)),
        epochs=int(s1_raw.get("epochs", 5)),
        batch_size=int(s1_raw.get("batch_size", 512)),
        lr=float(s1_raw.get("lr", 3e-3)),
        max_tokens=int(s1_raw.get("max_tokens", 512)),
        token_dropout=float(s1_raw.get("token_dropout", 0.05)),
        val_size=int(s1_raw.get("val_size", 4096)),
        apply_sif=bool(s1_raw.get("apply_sif", True)),
        sif_a=float(s1_raw.get("sif_a", 1e-3)),
        device=device,
        seed=seed,
        smoke_test=smoke_test,
    )
    _, _, s1_ckpt, _ = run_stage1(s1_cfg)

    checkpoint_registry: dict[str, Path] = {"s1": s1_ckpt}

    # Stage 2: One or more refinement branches (e.g. KL branch + Pretrain->InfoNCE branch)
    branches = cfg.get("stage2_branches") or {}
    if not branches and cfg.get("stage2"):
        branches = {"s2": cfg["stage2"]}

    for branch_name, b_raw in branches.items():
        b_out = Path(b_raw.get("output_dir", work_dir / f"stage2_{branch_name}"))
        b_cfg = Stage2Config(
            stage1_checkpoint=s1_ckpt,
            output_dir=b_out,
            pretrain_pairs_dir=_as_path(b_raw.get("pretrain_pairs_dir")),
            pretrain_q_col=b_raw.get("pretrain_q_col"),
            pretrain_d_col=b_raw.get("pretrain_d_col"),
            stage2a_batch_size=int(b_raw.get("stage2a_batch_size", 2048)),
            stage2a_lr=float(b_raw.get("stage2a_lr", 0.05)),
            stage2a_max_steps=int(b_raw.get("stage2a_max_steps", 2500)),
            stage2a_max_subset_rows=int(b_raw.get("stage2a_max_subset_rows", 500_000)),
            ft_source=b_raw.get("ft_source"),
            ft_format=b_raw.get("ft_format", "auto"),
            ft_config=b_raw.get("ft_config"),
            stage2b_epochs=int(b_raw.get("stage2b_epochs", 2)),
            stage2b_batch_size=int(b_raw.get("stage2b_batch_size", 1024)),
            stage2b_lr=float(b_raw["stage2b_lr"]) if b_raw.get("stage2b_lr") is not None else None,
            max_split_rows=int(b_raw.get("max_split_rows", 0)),
            max_negs_to_load=int(b_raw.get("max_negs_to_load", 10)),
            num_hard_negs=int(b_raw.get("num_hard_negs", 5)),
            kl_weight=float(b_raw.get("kl_weight", 0.0)),
            kl_teacher_temp=float(b_raw.get("kl_teacher_temp", 0.0)),
            mrl_dims=list(b_raw.get("mrl_dims", [128, 256, s1_cfg.dim])),
            max_tokens=int(b_raw.get("max_tokens", 512)),
            token_dropout=float(b_raw.get("token_dropout", 0.05)),
            sampling_alpha=float(b_raw.get("sampling_alpha", 0.5)),
            temperature=float(b_raw.get("temperature", 0.05)),
            fn_margin=float(b_raw.get("fn_margin", 0.05)),
            blend_alpha=float(b_raw.get("blend_alpha", 0.8)),
            device=device,
            seed=seed,
            smoke_test=smoke_test,
        )
        b_ckpts = run_stage2(b_cfg)
        if "stage2a" in b_ckpts:
            checkpoint_registry[f"{branch_name}_s2a"] = b_ckpts["stage2a"]
        if "stage2b" in b_ckpts:
            checkpoint_registry[f"{branch_name}_s2b"] = b_ckpts["stage2b"]

    # Stage 3: Norm-matched model soup
    soup_raw = cfg.get("soup")
    if soup_raw:
        recipes = soup_raw.get("recipes")
        if not recipes and soup_raw.get("weights"):
            recipes = {"final_soup": soup_raw["weights"]}
        if recipes:
            soup_out = Path(soup_raw.get("output_dir", work_dir / "final_soup"))
            out_path, _ = run_soup(
                checkpoints=checkpoint_registry,
                recipes=recipes,
                output_dir=soup_out,
                ref_key=soup_raw.get("ref_key", "s1"),
                eval_nanobeir=bool(soup_raw.get("eval_nanobeir", False)) and not smoke_test,
                batch_size=int(soup_raw.get("batch_size", 1024)),
                device=device,
            )
            return out_path

    return s1_ckpt
