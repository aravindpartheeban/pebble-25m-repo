---
base_model: MongoDB/voyage-4-nano
datasets:
  - ccde/fineweb-edu-sample-500K
  - lightonai/embeddings-pre-training-curated
  - lightonai/embeddings-fine-tuning-filtered-en
language:
  - en
license: apache-2.0
library_name: model2vec
pipeline_tag: feature-extraction
tags:
  - model2vec
  - static-embeddings
  - sentence-transformers
  - feature-extraction
  - retrieval
  - matryoshka
  - mrl
---

<!-- Place your banner image at ./assets/banner.png if publishing to Hugging Face -->

**`pebble-25m`** is a **24.6M-parameter English static embedding model** (`48,000 × 512d`, Matryoshka dims `[128, 256, 512]`) distilled from [`MongoDB/voyage-4-nano`](https://huggingface.co/MongoDB/voyage-4-nano) via Tokenlearn and refined with multi-domain contrastive pretraining, cross-encoder KL distillation, and norm-matched model souping.

Built for high-throughput CPU retrieval and hybrid BM25 + dense search pipelines, `pebble-25m` requires **no neural network forward pass at inference time**—encoding is a pure vocabulary lookup, mean pool, and L2 normalization (`43,854 docs/s` on CPU). On the 13-task **NanoBEIR-en** benchmark, `pebble-25m` achieves **51.99 nDCG@10** and **57.27 MRR@10**, outperforming `minishlab/blink-8M` (51.82 nDCG@10 / 56.29 MRR@10) and `minishlab/potion-base-32M` (46.18 nDCG@10 / 52.65 MRR@10), while matching `hotchpotch/bekko-static-embedding-v1@512` (52.17) at **8.3× fewer parameters**.

## Model Specifications

| Property | Value |
| --- | --- |
| **Architecture** | Static Token Embedding Lookup (`model2vec` / `StaticEmbedding`) |
| **Vocabulary** | Custom 48,000-token lowercase English WordPiece (`[PAD]` ID = `0`) |
| **Embedding Dimension** | `512` (Matryoshka truncatable to `256` or `128`) |
| **Parameters** | `24,576,000` (`48,000 × 512`) |
| **Checkpoint Size** | `93.8 MB` (`model.safetensors`, FP32) |
| **Teacher Model** | [`MongoDB/voyage-4-nano`](https://huggingface.co/MongoDB/voyage-4-nano) (Stage 1) + `mxbai-rerank-large-v2` scores (Stage 2b KL) |
| **Max Sequence Length** | `512` tokens |
| **Language** | English (`en`) |
| **License** | Apache-2.0 |

---

## Evaluation Results

All benchmarks below were run in standard `safetensors` floating-point format with L2-normalized embeddings (`normalize_embeddings=True`) using `sentence-transformers`'s `NanoBEIREvaluator`. Alongside the full **NanoBEIR-13** mean, we report **OOD-8 nDCG@10**—the mean across the 8 NanoBEIR tasks (`ArguAna`, `ClimateFEVER`, `DBPedia`, `NFCorpus`, `QuoraRetrieval`, `SCIDOCS`, `SciFact`, `Touche2020`) with zero training-split overlap in `lightonai/embeddings-fine-tuning-filtered-en`. CPU throughput was measured on a single machine using 64-sentence batches (`batch_size=64`).

### 1. Native Footprint, NanoBEIR-en & CPU Throughput

| Model | Params | Dim | Weights | NanoBEIR-13 nDCG@10 | OOD-8 nDCG@10 | MRR@10 | Recall@10 | Recall@100 | CPU docs/s |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pebble-25m`** | **24.6M** | **512** | **FP32 (93.8 MB)** | **51.99** | **50.07** | **57.27** | **57.29** | **83.39** | **43,854** |
| `minishlab/blink-8M` | 7.7M | 256 | FP32 (29.3 MB) | 51.82 | 50.25 | 56.29 | 58.38 | 83.81 | 49,248 |
| `minishlab/potion-base-32M` | 32.3M | 512 | FP32 (123.2 MB) | 46.18 | 46.88 | 52.65 | 50.00 | 79.62 | 45,899 |
| `hotchpotch/bekko-static-embedding-v1` | 204.8M | 1024 | FP16 (390.6 MB) | 53.10 | 52.64 | 58.42 | 58.79 | 85.08 | 25,944 |

### 2. Iso-Dimension Matryoshka Comparison (`512d` / `256d` / `128d`)

| Vector Dim | Model | NanoBEIR-13 nDCG@10 | OOD-8 nDCG@10 | MRR@10 | Recall@10 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `512d` | **`pebble-25m` (24.6M)** | **51.99** | 50.07 | **57.27** | **57.29** |
| `512d` | `potion-base-32M` (32.3M) | 46.18 | 46.88 | 52.65 | 50.00 |
| `512d` | `bekko-static-v1` (204.8M) | 52.17 | 52.10 | 57.69 | 57.55 |
| `256d` | **`pebble-25m` (24.6M)** | **50.50** | 48.94 | **55.95** | **56.39** |
| `256d` | `blink-8M` (7.7M) | 51.82 | 50.25 | 56.29 | 58.38 |
| `256d` | `bekko-static-v1` (204.8M) | 50.32 | 50.69 | 55.79 | 55.55 |
| `128d` | **`pebble-25m` (24.6M)** | **47.28** | 46.61 | **52.25** | **54.03** |
| `128d` | `bekko-static-v1` (204.8M) | 47.52 | 47.83 | 52.96 | 52.90 |

### 3. Per-Task NanoBEIR-en Breakdown (`nDCG@10 × 100`)

| Task | `pebble-25m` | `minishlab/blink-8M` | `minishlab/potion-base-32M` | `hotchpotch/bekko-static-embedding-v1` |
| :--- | :---: | :---: | :---: | :---: |
| `NanoArguAna` | 38.89 | 41.77 | 33.30 | 41.61 |
| `NanoClimateFEVER` | **34.11** | 30.80 | 27.75 | 34.63 |
| `NanoDBPedia` | **56.28** | 52.93 | 53.06 | 56.68 |
| `NanoFEVER` | 69.82 | 73.32 | 57.25 | 73.34 |
| `NanoFiQA2018` | **36.25** | 34.06 | 32.22 | 36.36 |
| `NanoHotpotQA` | **74.78** | 71.46 | 56.36 | 72.41 |
| `NanoMSMARCO` | 43.44 | 45.56 | 43.04 | 44.89 |
| `NanoNFCorpus` | 31.08 | 31.62 | 27.41 | 35.76 |
| `NanoNQ` | **51.05** | 47.25 | 36.41 | 42.12 |
| `NanoQuoraRetrieval` | **92.63** | 91.62 | 93.42 | 93.24 |
| `NanoSCIDOCS` | 31.32 | 34.77 | 28.46 | 35.70 |
| `NanoSciFact` | **66.65** | 66.05 | 60.30 | 71.19 |
| `NanoTouche2020` | 49.61 | 52.44 | 51.34 | 52.31 |
| **MEAN (All 13)** | **51.99** | 51.82 | 46.18 | 53.10 |
| **MEAN (OOD 8)** | 50.07 | 50.25 | 46.88 | 52.64 |

### 4. Training Progression & Ablations (`NanoBEIR-13`)

| Stage / Checkpoint | nDCG@10 | MRR@10 | Recall@10 | Description |
| :--- | :---: | :---: | :---: | :--- |
| Stage 1 (`Tokenlearn`) | 43.55 | 49.02 | 48.79 | 500k FineWeb-Edu distilled from `voyage-4-nano` (`1024d → 512d` PCA) + SIF |
| Stage 2a (`Pretrain Pairs`) | 47.00 | 51.85 | 52.99 | Multi-domain MRL InfoNCE on decontaminated LightOn pretrain slice |
| Stage 2b-v1 (`KL Blend 0.8/0.2`) | 49.81 | 55.60 | 54.94 | Hard-negative MRL InfoNCE + `mxbai-rerank-large-v2` KL distillation |
| Stage 2b-v2 (`S2a + InfoNCE Blend 0.8/0.2`) | 51.39 | 56.70 | **57.37** | Uncapped 9-split LightOn FT MRL InfoNCE initialized from Stage 2a |
| **Stage 3 (`pebble-25m` 4-Way Soup)** | **51.99** | **57.27** | 57.29 | Norm-matched soup (`0.10 S1 + 0.15 S2a + 0.25 S2b-v1 + 0.50 S2b-v2`) |

---

## Training Pipeline

`pebble-25m` was trained in four stages on a single local CUDA GPU:

1. **Stage 0 — Custom English WordPiece Tokenizer**: Trained a `48,000`-token lowercase WordPiece vocabulary (`BertNormalizer`, `BertPreTokenizer`, `min_frequency=3`, `[PAD]=0`, `[UNK]=1`) on 500k English passages from [`ccde/fineweb-edu-sample-500K`](https://huggingface.co/datasets/ccde/fineweb-edu-sample-500K).
2. **Stage 1 — Tokenlearn Distillation ([`MongoDB/voyage-4-nano`](https://huggingface.co/MongoDB/voyage-4-nano))**:
   - Encoded `500,000` English passages (clipped to `25–320` words, `max_seq_length=512`) using `MongoDB/voyage-4-nano` (`encode_document`, `1024d`, BF16/FP16 inference stored as FP16 Parquet).
   - Projected `1024d → 512d` via PCA fitted on `100,000` teacher vectors and trained a `48,000 × 512` `EmbeddingBag` + foldable linear projection with cosine distance loss (`5` epochs, `batch_size=512`, `lr=3e-3`, cosine decay with 5% warmup, `5%` token dropout), followed by Smooth Inverse Frequency (SIF, $a = 10^{-3}$) rescaling.
3. **Stage 2a & 2b — Multi-Domain Contrastive Pretraining + Hard-Negative KL Distillation**:
   - **Stage 2a**: Streamed `2,500` steps (`batch_size=2048`, `~5.12M` pairs, `lr=0.05`) across ~31 decontaminated domains from [`lightonai/embeddings-pre-training-curated`](https://huggingface.co/datasets/lightonai/embeddings-pre-training-curated) (shard `00000` per subset, excluding direct BEIR/MSMARCO/Quora overlap) using temperature-balanced sampling ($\alpha = 0.5$), Matryoshka dims `[128, 256, 512]`, and false-negative masking (`margin = 0.05`, $\tau = 0.05$).
   - **Stage 2b**: Trained two complementary fine-tuning branches across all 9 English splits (`fever_en`, `fiqa_en`, `hotpotqa_en`, `miracl_en`, `mldr_en`, `msmarco_en`, `nq_en`, `squadv2_en`, `trivia_en` — ~1.57M queries with 1 positive + 5 sampled hard negatives per step) of [`lightonai/embeddings-fine-tuning-filtered-en`](https://huggingface.co/datasets/lightonai/embeddings-fine-tuning-filtered-en):
     - *Branch v1*: Joint MRL Masked InfoNCE ($\tau = 0.05$) + listwise KL distillation (`kl_weight = 1.0`) against the dataset's `mxbai-rerank-large-v2` cross-encoder `rerank_scores` (`batch_size=512`, `lr=0.05`, `2` epochs).
     - *Branch v2*: Uncapped 9-split MRL Masked InfoNCE (`batch_size=1024`, `lr=0.015`, `2` epochs, `kl_weight = 0.0`) initialized from Stage 2a.
4. **Stage 3 — Norm-Matched Model Soup**:
   - Rescaled each checkpoint's token matrix to match the median L2 norm of Stage 1 (`[PAD]` pinned to zero) and blended `0.10 × Stage 1 + 0.15 × Stage 2a + 0.25 × Stage 2b (v1) + 0.50 × Stage 2b (v2)`.

---

## Quickstart Usage

### 1. Using `model2vec` (Fastest CPU Inference)

```python
from model2vec import StaticModel

model = StaticModel.from_pretrained("waypoint-ML/pebble-25m")

sentences = [
    "How do I rotate production API keys?",
    "Steps for rotating and revoking live service credentials in production.",
    "Sourdough fermentation relies on wild yeast and lactic acid bacteria.",
]
embeddings = model.encode(sentences)  # shape: (3, 512), L2-normalized
print(embeddings @ embeddings.T)
```

### 2. Using `sentence-transformers` (with Optional Matryoshka Truncation)

```python
from sentence_transformers import SentenceTransformer

# Full 512d (or pass truncate_dim=256 / truncate_dim=128 for smaller vectors)
model = SentenceTransformer("waypoint-ML/pebble-25m", truncate_dim=512)

queries = ["Why does a Kubernetes pod enter CrashLoopBackOff?"]
docs = [
    "CrashLoopBackOff occurs when a container repeatedly crashes after starting.",
    "Configure TLS certificates for an Nginx reverse proxy.",
]
q_emb = model.encode_query(queries)
d_emb = model.encode_document(docs)
print(model.similarity(q_emb, d_emb))
```

---

## Limitations, Risks, and Bias

- **English-Only**: Trained exclusively on English web and retrieval corpora; non-English text will produce degraded representations.
- **Bag-of-Tokens Semantics**: Static embeddings average token vectors without contextual self-attention. Negation (*"enable SSO"* vs. *"disable SSO"*), multi-hop reasoning, and exact identifier/code matching (`PROJ-1842`, error codes) are best handled by pairing `pebble-25m` with lexical **BM25** search (e.g., via Reciprocal Rank Fusion) and/or a lightweight cross-encoder reranker.
- **Domain Coverage**: Distilled from `500k` `FineWeb-Edu` passages and LightOn retrieval datasets; specialized citation-graph and biomedical corpora (`NanoSCIDOCS`: 31.32, `NanoNFCorpus`: 31.08) trail specialized dense transformers.
- **Inherited Corpus & Teacher Biases**: Distilled from `MongoDB/voyage-4-nano` and public web/QA datasets, which may carry web-text skew or factual inaccuracies.

## Acknowledgements

Built with [MinishLab Model2Vec / Tokenlearn](https://github.com/MinishLab/model2vec), [`MongoDB/voyage-4-nano`](https://huggingface.co/MongoDB/voyage-4-nano), and LightOn's curated retrieval datasets.
