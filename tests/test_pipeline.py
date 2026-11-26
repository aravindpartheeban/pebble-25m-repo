"""Offline end-to-end tests for all Pebble pipeline stages using synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from model2vec import StaticModel

from pebble_pipeline.featurize import write_fp16_parquet
from pebble_pipeline.pipeline import run_from_config
from pebble_pipeline.soup import norm_matched_soup
from pebble_pipeline.stage1 import Stage1Config, run_stage1
from pebble_pipeline.stage2 import Stage2Config, run_stage2


def _make_synthetic_teacher_shards(out_dir: Path, n_rows: int = 80, dim: int = 64) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    vocab_words = [
        "kubernetes", "pod", "container", "nginx", "tls", "certificate", "neural",
        "gradient", "descent", "bond", "yield", "interest", "rate", "sourdough",
        "yeast", "fermentation", "database", "query", "index", "retrieval",
    ]
    texts = [
        " ".join(rng.choice(vocab_words, size=14, replace=True))
        for _ in range(n_rows)
    ]
    embs = rng.normal(size=(n_rows, dim)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    write_fp16_parquet(out_dir / "shard0_0000.parquet", texts, embs)
    return out_dir


def _make_synthetic_pairs(out_dir: Path, n_rows: int = 64) -> Path:
    sub = out_dir / "domain_a"
    sub.mkdir(parents=True, exist_ok=True)
    queries = [f"how does topic {i} work in distributed systems" for i in range(n_rows)]
    docs = [f"explanation of topic {i} covering kubernetes pods and database indexing" for i in range(n_rows)]
    tbl = pa.table({"query": pa.array(queries), "document": pa.array(docs)})
    pq.write_table(tbl, sub / "pairs.parquet")
    return out_dir


def _make_synthetic_triplets(out_path: Path, n_rows: int = 48) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n_rows):
            row = {
                "query": f"what causes error {i} in container deployment",
                "positive": f"container deployment error {i} happens when image pull or probe fails",
                "negatives": [
                    f"sourdough fermentation recipe step {i} with wild yeast",
                    f"bond yield curve analysis scenario {i} under rate hikes",
                    f"nginx tls upstream timeout configuration {i}",
                    f"neural network gradient descent update rule {i}",
                ],
                "rerank_scores": [0.95, 0.05, 0.04, 0.25, 0.02],
            }
            f.write(json.dumps(row) + "\n")
    return out_path


def test_stage1_stage2_and_soup_end_to_end(tmp_path: Path) -> None:
    teacher_dir = _make_synthetic_teacher_shards(tmp_path / "teacher", n_rows=80, dim=64)
    pairs_dir = _make_synthetic_pairs(tmp_path / "pairs", n_rows=64)
    triplets_file = _make_synthetic_triplets(tmp_path / "triplets.jsonl", n_rows=48)

    # Stage 0 + 1 with PCA (64 -> 32)
    s1_vecs, tok, s1_ckpt, val_cos = run_stage1(
        Stage1Config(
            stage1_dir=teacher_dir,
            output_dir=tmp_path / "out_s1",
            vocab_size=256,
            min_frequency=1,
            dim=32,
            epochs=1,
            batch_size=16,
            apply_sif=True,
            device="cpu",
            smoke_test=True,
        )
    )
    assert s1_vecs.shape[1] == 32
    assert np.isfinite(s1_vecs).all()
    assert -1.0 <= val_cos <= 1.0
    assert (s1_ckpt / "model.safetensors").exists()

    # Stage 2a + 2b (with KL distillation)
    ckpts = run_stage2(
        Stage2Config(
            stage1_checkpoint=s1_ckpt,
            output_dir=tmp_path / "out_s2",
            pretrain_pairs_dir=pairs_dir,
            ft_source=str(triplets_file),
            ft_format="triplet",
            num_hard_negs=3,
            kl_weight=0.5,
            mrl_dims=[16, 32],
            device="cpu",
            smoke_test=True,
        )
    )
    assert "stage2a" in ckpts and (ckpts["stage2a"] / "model.safetensors").exists()
    assert "stage2b" in ckpts and (ckpts["stage2b"] / "model.safetensors").exists()
    assert "final_blended" in ckpts and (ckpts["final_blended"] / "model.safetensors").exists()

    # Verify pad row stays zeroed in Stage 2b output
    m_s2b = StaticModel.from_pretrained(str(ckpts["stage2b"]))
    pad_id = tok.token_to_id("[PAD]")
    assert pad_id is not None
    assert np.allclose(np.asarray(m_s2b.embedding)[pad_id], 0.0)

    # Norm-matched 3-way soup
    m_s2a = StaticModel.from_pretrained(str(ckpts["stage2a"]))
    souped = norm_matched_soup(
        {
            "s1": s1_vecs,
            "s2a": np.asarray(m_s2a.embedding, dtype=np.float32),
            "s2b": np.asarray(m_s2b.embedding, dtype=np.float32),
        },
        {"s1": 0.2, "s2a": 0.3, "s2b": 0.5},
        ref_key="s1",
        pad_id=pad_id,
    )
    assert souped.shape == s1_vecs.shape
    assert np.allclose(souped[pad_id], 0.0)


def test_stage1_skips_pca_when_dims_match(tmp_path: Path) -> None:
    teacher_dir = _make_synthetic_teacher_shards(tmp_path / "teacher_32", n_rows=60, dim=32)
    s1_vecs, _, _, _ = run_stage1(
        Stage1Config(
            stage1_dir=teacher_dir,
            output_dir=tmp_path / "out_nopca",
            vocab_size=200,
            min_frequency=1,
            dim=32,
            epochs=1,
            batch_size=16,
            device="cpu",
            smoke_test=True,
        )
    )
    assert s1_vecs.shape[1] == 32


def test_config_pipeline_runner_smoke(tmp_path: Path) -> None:
    teacher_dir = _make_synthetic_teacher_shards(tmp_path / "teacher", n_rows=80, dim=64)
    pairs_dir = _make_synthetic_pairs(tmp_path / "pairs", n_rows=64)
    triplets_file = _make_synthetic_triplets(tmp_path / "triplets.jsonl", n_rows=48)

    cfg = {
        "seed": 7,
        "device": "cpu",
        "output_dir": str(tmp_path / "pipeline_out"),
        "stage1": {
            "stage1_dir": str(teacher_dir),
            "vocab_size": 256,
            "min_frequency": 1,
            "dim": 32,
        },
        "stage2_branches": {
            "kl": {
                "ft_source": str(triplets_file),
                "ft_format": "triplet",
                "num_hard_negs": 3,
                "kl_weight": 0.5,
                "mrl_dims": [16, 32],
            },
            "infonce": {
                "pretrain_pairs_dir": str(pairs_dir),
                "ft_source": str(triplets_file),
                "ft_format": "triplet",
                "num_hard_negs": 3,
                "kl_weight": 0.0,
                "mrl_dims": [16, 32],
            },
        },
        "soup": {
            "output_dir": str(tmp_path / "pipeline_out" / "final_soup"),
            "ref_key": "s1",
            "weights": {
                "s1": 0.10,
                "infonce_s2a": 0.15,
                "kl_s2b": 0.25,
                "infonce_s2b": 0.50,
            },
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    final_dir = run_from_config(cfg_path, smoke_test=True)
    model = StaticModel.from_pretrained(str(final_dir))
    embs = model.encode(["container deployment error", "sourdough fermentation"])
    assert embs.shape == (2, 32)
    assert np.isfinite(embs).all()


def test_stage1_persists_max_tokens(tmp_path: Path) -> None:
    """The training max_tokens must be stored as the model's inference max_length."""
    teacher_dir = _make_synthetic_teacher_shards(tmp_path / "teacher_mt", n_rows=60, dim=32)
    _, _, s1_ckpt, _ = run_stage1(
        Stage1Config(
            stage1_dir=teacher_dir,
            output_dir=tmp_path / "out_mt",
            vocab_size=200,
            min_frequency=1,
            dim=32,
            epochs=1,
            batch_size=16,
            max_tokens=256,
            device="cpu",
            smoke_test=True,
        )
    )
    assert StaticModel.from_pretrained(str(s1_ckpt)).max_length == 256


def test_stage2_kl_weight_requires_scores(tmp_path: Path) -> None:
    """kl_weight > 0 with no aligned rerank_scores must fail fast instead of silently skipping KL."""
    teacher_dir = _make_synthetic_teacher_shards(tmp_path / "teacher_kl", n_rows=60, dim=32)
    _, _, s1_ckpt, _ = run_stage1(
        Stage1Config(
            stage1_dir=teacher_dir,
            output_dir=tmp_path / "out_kl",
            vocab_size=200,
            min_frequency=1,
            dim=32,
            epochs=1,
            batch_size=16,
            device="cpu",
            smoke_test=True,
        )
    )
    triplets = tmp_path / "triplets_noscores.jsonl"
    with triplets.open("w", encoding="utf-8") as f:
        for i in range(16):
            row = {
                "query": f"query {i}",
                "positive": f"positive {i}",
                "negatives": [f"neg {i}a", f"neg {i}b", f"neg {i}c"],
            }
            f.write(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="rerank_scores"):
        run_stage2(
            Stage2Config(
                stage1_checkpoint=s1_ckpt,
                output_dir=tmp_path / "out_kl_fail",
                ft_source=str(triplets),
                ft_format="triplet",
                num_hard_negs=2,
                kl_weight=1.0,
                mrl_dims=[16, 32],
                device="cpu",
                smoke_test=True,
            )
        )
