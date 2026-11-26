"""Command-line entrypoints for all Pebble pipeline stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import run_benchmark
from .featurize import FeaturizeConfig, run_featurize
from .pipeline import run_from_config
from .soup import run_soup
from .stage1 import Stage1Config, run_stage1
from .stage2 import Stage2Config, run_stage2


def featurize_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Extract FP16 teacher embeddings into Parquet shards.")
    p.add_argument("--model-id", type=str, default="MongoDB/voyage-4-nano")
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dim", type=int, default=1024, help="Optional Matryoshka teacher dim (0 = full model dim).")
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--encode-mode", choices=["auto", "document", "encode"], default="auto")

    p.add_argument("--dataset", type=str, default="ccde/fineweb-edu-sample-500K", help="HF dataset repo ID.")
    p.add_argument("--dataset-config", type=str, default=None, help="Optional HF dataset config/subset.")
    p.add_argument("--dataset-split", type=str, default="train")
    p.add_argument("--text-column", type=str, default="text")
    p.add_argument("--local-files", type=str, nargs="*", default=None, help="Local Parquet/JSONL/TXT files instead of HF.")

    p.add_argument("--output-dir", type=Path, default=Path("./data/teacher_features"))
    p.add_argument("--target-rows", type=int, default=500_000)
    p.add_argument("--rows-per-file", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--min-words", type=int, default=25)
    p.add_argument("--max-words", type=int, default=320)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--push-to-hub", type=str, default="", help="Optional HF dataset repo ID to upload shards.")
    p.add_argument("--smoke-test", action="store_true", help="Encode 500 passages to verify setup.")
    a = p.parse_args(argv)

    run_featurize(
        FeaturizeConfig(
            model_id=a.model_id,
            trust_remote_code=a.trust_remote_code,
            dim=None if a.dim <= 0 else a.dim,
            max_seq_length=a.max_seq_length,
            encode_mode=a.encode_mode,
            dataset=a.dataset,
            dataset_config=a.dataset_config,
            dataset_split=a.dataset_split,
            text_column=a.text_column,
            local_files=a.local_files,
            output_dir=a.output_dir,
            target_rows=a.target_rows,
            rows_per_file=a.rows_per_file,
            batch_size=a.batch_size,
            min_words=a.min_words,
            max_words=a.max_words,
            shard_id=a.shard_id,
            num_shards=a.num_shards,
            device=a.device,
            push_to_hub=a.push_to_hub,
            smoke_test=a.smoke_test,
        )
    )


def stage1_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stage 0 (WordPiece Tokenizer) + Stage 1 (Tokenlearn + SIF).")
    p.add_argument("--stage1-dir", type=Path, required=True, help="Directory of teacher feature Parquet shards.")
    p.add_argument("--output-dir", type=Path, default=Path("./outputs/run"))
    p.add_argument("--text-col", type=str, default="text")
    p.add_argument("--embedding-col", type=str, default="embedding")

    p.add_argument("--tokenizer-path", type=Path, default=None, help="Optional pre-trained tokenizers.Tokenizer JSON.")
    p.add_argument("--vocab-size", type=int, default=48_000)
    p.add_argument("--min-frequency", type=int, default=3)
    p.add_argument("--tokenizer-texts", type=int, default=500_000)
    p.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strip-accents", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--pca-fit-samples", type=int, default=100_000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--token-dropout", type=float, default=0.05)
    p.add_argument("--val-size", type=int, default=4096)
    p.add_argument("--apply-sif", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sif-a", type=float, default=1e-3)

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true")
    a = p.parse_args(argv)

    run_stage1(
        Stage1Config(
            stage1_dir=a.stage1_dir,
            output_dir=a.output_dir,
            text_col=a.text_col,
            embedding_col=a.embedding_col,
            tokenizer_path=a.tokenizer_path,
            vocab_size=a.vocab_size,
            min_frequency=a.min_frequency,
            tokenizer_texts=a.tokenizer_texts,
            lowercase=a.lowercase,
            strip_accents=a.strip_accents,
            dim=a.dim,
            pca_fit_samples=a.pca_fit_samples,
            epochs=a.epochs,
            batch_size=a.batch_size,
            lr=a.lr,
            max_tokens=a.max_tokens,
            token_dropout=a.token_dropout,
            val_size=a.val_size,
            apply_sif=a.apply_sif,
            sif_a=a.sif_a,
            device=a.device,
            seed=a.seed,
            smoke_test=a.smoke_test,
        )
    )


def stage2_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stage 2a (Pairs Pretraining) + Stage 2b (Hard-Neg MRL InfoNCE + KL).")
    p.add_argument("--stage1-checkpoint", type=Path, required=True, help="Saved stage1_tokenlearn directory.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true")

    # Stage 2a
    p.add_argument("--pretrain-pairs-dir", type=Path, default=None, help="Optional Stage 2a Parquet pairs directory.")
    p.add_argument("--pretrain-q-col", type=str, default=None)
    p.add_argument("--pretrain-d-col", type=str, default=None)
    p.add_argument("--stage2a-batch-size", type=int, default=2048)
    p.add_argument("--stage2a-lr", type=float, default=0.05)
    p.add_argument("--stage2a-max-steps", type=int, default=2500)
    p.add_argument("--stage2a-max-subset-rows", type=int, default=500_000)

    # Stage 2b
    p.add_argument("--ft-source", type=str, default="lightonai/embeddings-fine-tuning-filtered-en")
    p.add_argument("--ft-format", choices=["auto", "lighton", "triplet"], default="auto")
    p.add_argument("--ft-config", type=str, default=None)
    p.add_argument("--stage2b-epochs", type=int, default=2)
    p.add_argument("--stage2b-batch-size", type=int, default=1024)
    p.add_argument("--stage2b-lr", type=float, default=None)
    p.add_argument("--max-split-rows", type=int, default=0)
    p.add_argument("--max-negs-to-load", type=int, default=10)
    p.add_argument("--num-hard-negs", type=int, default=5)
    p.add_argument("--kl-weight", type=float, default=0.0)
    p.add_argument("--kl-teacher-temp", type=float, default=0.0)

    # Shared
    p.add_argument("--mrl-dims", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--token-dropout", type=float, default=0.05)
    p.add_argument("--sampling-alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--fn-margin", type=float, default=0.05)
    p.add_argument("--blend-alpha", type=float, default=0.8)
    a = p.parse_args(argv)

    run_stage2(
        Stage2Config(
            stage1_checkpoint=a.stage1_checkpoint,
            output_dir=a.output_dir,
            pretrain_pairs_dir=a.pretrain_pairs_dir,
            pretrain_q_col=a.pretrain_q_col,
            pretrain_d_col=a.pretrain_d_col,
            stage2a_batch_size=a.stage2a_batch_size,
            stage2a_lr=a.stage2a_lr,
            stage2a_max_steps=a.stage2a_max_steps,
            stage2a_max_subset_rows=a.stage2a_max_subset_rows,
            ft_source=a.ft_source or None,
            ft_format=a.ft_format,
            ft_config=a.ft_config,
            stage2b_epochs=a.stage2b_epochs,
            stage2b_batch_size=a.stage2b_batch_size,
            stage2b_lr=a.stage2b_lr,
            max_split_rows=a.max_split_rows,
            max_negs_to_load=a.max_negs_to_load,
            num_hard_negs=a.num_hard_negs,
            kl_weight=a.kl_weight,
            kl_teacher_temp=a.kl_teacher_temp,
            mrl_dims=a.mrl_dims,
            max_tokens=a.max_tokens,
            token_dropout=a.token_dropout,
            sampling_alpha=a.sampling_alpha,
            temperature=a.temperature,
            fn_margin=a.fn_margin,
            blend_alpha=a.blend_alpha,
            device=a.device,
            seed=a.seed,
            smoke_test=a.smoke_test,
        )
    )


def _parse_kv_paths(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        name, path = item.split("=", 1)
        out[name.strip()] = Path(path.strip())
    return out


def _parse_recipes(items: list[str]) -> dict[str, dict[str, float]]:
    recipes: dict[str, dict[str, float]] = {}
    for idx, spec in enumerate(items):
        if ":" in spec:
            r_name, weights_str = spec.split(":", 1)
        else:
            r_name, weights_str = f"recipe_{idx}", spec
        w_map: dict[str, float] = {}
        for part in weights_str.split(","):
            k, v = part.split("=", 1)
            w_map[k.strip()] = float(v.strip())
        recipes[r_name.strip()] = w_map
    return recipes


def soup_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stage 3: Norm-matched linear model soup across checkpoints.")
    p.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Named checkpoint in the form name=path (repeatable).",
    )
    p.add_argument(
        "--recipe",
        action="append",
        default=[],
        help="Recipe in the form name:k1=w1,k2=w2 (repeatable).",
    )
    p.add_argument("--recipe-file", type=Path, default=None, help="Optional JSON file mapping recipe names to weights.")
    p.add_argument("--ref-key", type=str, default=None, help="Reference checkpoint key for norm matching.")
    p.add_argument("--output-dir", type=Path, default=Path("./outputs/soup_best"))
    p.add_argument("--eval-nanobeir", action="store_true", help="Evaluate recipes on NanoBEIR-en and save winner.")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", type=str, default=None)
    a = p.parse_args(argv)

    checkpoints = _parse_kv_paths(a.checkpoint)
    recipes: dict[str, dict[str, float]] = {}
    if a.recipe_file:
        with a.recipe_file.open("r", encoding="utf-8") as f:
            recipes.update(json.load(f))
    if a.recipe:
        recipes.update(_parse_recipes(a.recipe))
    if not recipes:
        equal_w = 1.0 / len(checkpoints)
        recipes["equal_soup"] = {k: equal_w for k in checkpoints}

    run_soup(
        checkpoints=checkpoints,
        recipes=recipes,
        output_dir=a.output_dir,
        ref_key=a.ref_key,
        eval_nanobeir=a.eval_nanobeir,
        batch_size=a.batch_size,
        device=a.device,
    )


def eval_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate static embedding checkpoints on NanoBEIR / MNanoBEIR.")
    p.add_argument("--models", type=str, nargs="*", default=None, help="Local paths or HF model IDs (supports @dim).")
    p.add_argument("--run-dir", type=Path, default=None, help="Run directory containing stage checkpoints.")
    p.add_argument("--sweep-alphas", type=str, default="", help="Comma-separated alpha values for Stage 1+2 blends.")
    p.add_argument("--langs", type=str, default="en", help="Comma-separated language codes (e.g. 'en' or 'en,de,fr').")
    p.add_argument("--mrl-dims", type=int, nargs="*", default=None, help="Optional Matryoshka dims to evaluate.")
    p.add_argument("--no-speed", action="store_true", help="Skip CPU throughput benchmark.")
    p.add_argument("--cpu-iters", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("./outputs/eval"))
    a = p.parse_args(argv)

    alphas = [float(x.strip()) for x in a.sweep_alphas.split(",") if x.strip()] if a.sweep_alphas.strip() else None
    langs = [x.strip() for x in a.langs.split(",") if x.strip()]
    run_benchmark(
        models=a.models,
        run_dir=a.run_dir,
        sweep_alphas=alphas,
        langs=langs,
        mrl_dims=a.mrl_dims,
        measure_speed=not a.no_speed,
        cpu_iters=a.cpu_iters,
        batch_size=a.batch_size,
        device=a.device,
        output_dir=a.output_dir,
    )


def pipeline_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run the full Pebble static embedding pipeline from a YAML config.")
    p.add_argument("--config", type=Path, required=True, help="Path to YAML pipeline config.")
    p.add_argument("--smoke-test", action="store_true", help="Run a fast end-to-end smoke pass.")
    a = p.parse_args(argv)
    run_from_config(a.config, smoke_test=a.smoke_test)
