# Pebble Static Embedding Pipeline

A clean, modular, dataset-agnostic pipeline for distilling dense neural retrieval models into ultra-fast **static token embedding models** (`model2vec` / `SentenceTransformer` `StaticEmbedding`), refined with Matryoshka Representation Learning (MRL), hard-negative contrastive training, optional cross-encoder KL distillation, and norm-matched model souping.

`waypoint-ML/pebble-25m` ([model card](MODEL_CARD.md)) was trained using this 4-stage architecture. You can swap in **any teacher model, text corpus, positive-pair dataset, hard-negative dataset, vocabulary size, or embedding dimension** to build your own static embedding model.

---

## Architecture Overview

```text
+----------------------------------------------------+
|  Raw Text Corpus                                   |
|  featurize.py (Teacher SentenceTransformer)        |
|  -> FP16 Parquet Shards                            |
+----------------------------------------------------+
                          |
                          v
+----------------------------------------------------+
|  [Stage 0] Custom WordPiece Tokenizer              |
|  (or bring your own tokenizer.json)                |
+----------------------------------------------------+
                          |
                          v
+----------------------------------------------------+
|  [Stage 1] Tokenlearn Distillation                 |----+
|  EmbeddingBag + foldable linear proj + PCA + SIF   |    |
+----------------------------------------------------+    |
                          |                               |
                          v                               |
+----------------------------------------------------+    |
|  [Stage 2a] Contrastive Pairs                      |    |
|  Streaming temperature-balanced InfoNCE            |    |
|  + optional listwise KL                            |    |
+----------------------------------------------------+    |
                          |                               |
                          v                               |
+----------------------------------------------------+    |
|  [Stage 2b] Hard-Negative MRL                      |<---+
+----------------------------------------------------+    |
                          |                               |
                          v                               |
+----------------------------------------------------+    |
|  [Stage 3] Norm-Matched N-Way Model Soup           |<---+
+----------------------------------------------------+
                          |
                          v
+----------------------------------------------------+
|  Exported model2vec / SentenceTransformer          |
|  StaticModel                                       |
+----------------------------------------------------+
```

### Key Design Choices
- **Foldable Stage-1 Projection (`TokenlearnStudent`)**: Trains an `EmbeddingBag` + linear projection head against PCA-projected (or native-dimension) teacher vectors with cosine distance loss and token dropout, then folds the linear matrix into the token lookup table and applies Smooth Inverse Frequency (SIF) weighting.
- **False-Negative Masked MRL InfoNCE**: Optimizes nested Matryoshka prefixes (e.g. `[128, 256, 512]`) simultaneously while masking in-batch negatives whose similarity exceeds `pos_sim + fn_margin`.
- **Optional Listwise Cross-Encoder KL Distillation**: Aligns student candidate distributions over `[positive, *hard_negatives]` with reranker scores (`rerank_scores`) using auto-calibrated teacher temperature.
- **Norm-Matched Model Souping**: Rescales each checkpoint's non-pad token vectors to a shared median L2 norm before linear averaging, combining generalization from Stage 1/2a with sharp ranking from Stage 2b without extra training.

---

## Installation

```bash
git clone https://github.com/waypoint-ML/pebble-25m.git
cd pebble-25m
pip install -e .
# or: pip install -r requirements.txt
```

Requires Python `>=3.10` and PyTorch (`2.4+` recommended; CUDA optional for Stage 1/2 training, CPU supported for inference and evaluation).

---

## Repository Structure

```text
├── configs/
│   └── pebble_25m.yaml        # Example preset config (reproduces the pebble-25m recipe)
├── scripts/
│   ├── featurize.py           # Extract FP16 teacher embeddings to Parquet shards
│   ├── stage1.py              # Stage 0 (Tokenizer) + Stage 1 (Tokenlearn + PCA + SIF)
│   ├── stage2.py              # Stage 2a (Pairs Pretraining) + Stage 2b (Hard-Neg MRL + KL)
│   ├── soup.py                # Stage 3 (N-way norm-matched model soup & sweep)
│   ├── eval.py                # Standardized NanoBEIR / MNanoBEIR / Matryoshka / CPU speed eval
│   └── run_pipeline.py        # End-to-end YAML config runner
├── src/pebble_pipeline/       # Core library modules
└── tests/                     # Offline synthetic unit & smoke tests
```

---

## Usage Guide (Bring Your Own Model & Data)

### 1. Extract Teacher Features (`featurize.py`)
Encode any Hugging Face dataset or local `.parquet` / `.jsonl` / `.txt` files using any `SentenceTransformer` teacher:

```bash
# From a Hugging Face dataset
python scripts/featurize.py \
  --model-id MongoDB/voyage-4-nano \
  --dim 1024 \
  --dataset ccde/fineweb-edu-sample-500K \
  --text-column text \
  --target-rows 500000 \
  --output-dir ./data/teacher_features

# Or from local files
python scripts/featurize.py \
  --model-id BAAI/bge-m3 \
  --dim 0 \
  --local-files ./my_corpus/*.parquet \
  --text-column body \
  --output-dir ./data/teacher_features
```

Add `--smoke-test` to test on 500 passages, or `--shard-id 0 --num-shards 2` for multi-worker sharding.

### 2. Stage 0 & Stage 1: Tokenizer + Tokenlearn Distillation (`stage1.py`)
Train a custom WordPiece tokenizer (or pass `--tokenizer-path` to reuse one) and distill the teacher features into a static embedding matrix:

```bash
python scripts/stage1.py \
  --stage1-dir ./data/teacher_features \
  --output-dir ./outputs/my_model \
  --vocab-size 48000 \
  --dim 512 \
  --epochs 5 \
  --batch-size 512 \
  --lr 3e-3
```
If your teacher dimension equals `--dim`, PCA is automatically skipped.

### 3. Stage 2: Contrastive Pretraining & Hard-Negative Refinement (`stage2.py`)
Refine the Stage-1 checkpoint using optional Stage-2a positive-pair shards and Stage-2b hard negatives:

- **Stage 2a data format** (`--pretrain-pairs-dir`): Any directory of `.parquet` files (nested subdirectories or root shards). Automatically detects `query`/`anchor`/`question`/`text` and `document`/`positive`/`pos`/`passage`/`answer` columns, or override via `--pretrain-q-col` / `--pretrain-d-col`.
- **Stage 2b data format** (`--ft-source`): Supports both:
  1. **Generic triplets** (Parquet, JSONL, or HF Hub): columns `query`, `positive`, `negatives` (`list[str]`), and optional `rerank_scores` (`list[float]` aligned with `[positive, *negatives]`).
  2. **LightOn 3-config schema** (`scores`, `documents`, `queries`), e.g. `lightonai/embeddings-fine-tuning-filtered-en`.

```bash
# Branch A: Hard-negative MRL InfoNCE + Cross-Encoder KL distillation
python scripts/stage2.py \
  --stage1-checkpoint ./outputs/my_model/stage1_tokenlearn \
  --output-dir ./outputs/my_model_kl \
  --ft-source ./data/my_triplets.parquet \
  --kl-weight 1.0 \
  --mrl-dims 128 256 512

# Branch B: Stage 2a multi-domain pairs -> Stage 2b pure hard-negative InfoNCE
python scripts/stage2.py \
  --stage1-checkpoint ./outputs/my_model/stage1_tokenlearn \
  --pretrain-pairs-dir ./data/pair_shards \
  --output-dir ./outputs/my_model_infonce \
  --ft-source ./data/my_triplets.parquet \
  --kl-weight 0.0 \
  --mrl-dims 128 256 512
```

### 4. Stage 3: Norm-Matched Model Soup (`soup.py`)
Blend any number of checkpoints sharing the same Stage-1 initialization and tokenizer:

```bash
python scripts/soup.py \
  --checkpoint s1=./outputs/my_model/stage1_tokenlearn \
  --checkpoint s2a=./outputs/my_model_infonce/stage2a_pretrain \
  --checkpoint kl=./outputs/my_model_kl/stage2_contrastive \
  --checkpoint infonce=./outputs/my_model_infonce/stage2_contrastive \
  --recipe "4way:s1=0.10,s2a=0.15,kl=0.25,infonce=0.50" \
  --output-dir ./outputs/my_model_final \
  --eval-nanobeir
```

### 5. One-Command Config Execution (`run_pipeline.py`)
See [`configs/pebble_25m.yaml`](configs/pebble_25m.yaml) for a full multi-branch configuration:

```bash
python scripts/run_pipeline.py --config configs/pebble_25m.yaml
```

### 6. Standardized Evaluation (`eval.py`)
Evaluate any local checkpoint or Hugging Face model (with `@dim` Matryoshka truncation, CPU throughput measurement, and optional `--sweep-alphas`):

```bash
python scripts/eval.py \
  --models ./outputs/my_model_final minishlab/blink-8M hotchpotch/bekko-static-embedding-v1@512 \
  --langs en \
  --mrl-dims 512 256 128
```

---

## Reference Preset: `pebble-25m`

Using the architecture in this repository (`48,000 × 512` WordPiece, `MongoDB/voyage-4-nano` Stage-1 teacher, `MRL=[128, 256, 512]`, 4-way norm-matched soup), `waypoint-ML/pebble-25m` achieved **51.99 nDCG@10** across NanoBEIR-en (13 tasks) at **43,854 docs/s** on CPU. Full evaluation tables and model details are documented in [MODEL_CARD.md](MODEL_CARD.md).

---

## Testing

Run the offline synthetic end-to-end test suite (no internet or GPU required):

```bash
pytest -v
```

## License

Apache-2.0 - see [LICENSE](LICENSE).
