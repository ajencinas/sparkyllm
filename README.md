# SparkLLM

A from-scratch GPT-style language model built for learning and experimentation.

## Model

- **Architecture:** Transformer decoder (SimpleGPT) with SwiGLU feed-forward, Flash Attention, Pre-LayerNorm
- **Size:** ~650M parameters (24 layers, 1280 embed dim, 20 heads)
- **Context:** 768 tokens
- **Tokenizer:** Byte-Level BPE (32K vocab)
- **Training:** AdamW with cosine LR schedule, BFloat16 mixed precision, torch.compile

## Project Structure

```
sparkyllm/
├── pretrain/                           # Pre-training pipeline (Colab notebooks)
│   ├── download_c4.ipynb               # Download C4 dataset from HuggingFace
│   ├── download_code.ipynb             # Download code datasets
│   ├── download_wikipedia.ipynb        # Download full English Wikipedia
│   ├── tokenizer_pipeline.ipynb        # Train tokenizer + encode into shards
│   └── pre_train_mar23.ipynb           # Main training loop (sharded)
├── sft/                                # Supervised Fine-Tuning
│   ├── download_alpaca.ipynb           # Download Alpaca instruction dataset
│   ├── tokenize_sft.ipynb              # Tokenize SFT data
│   └── sft_alpaca.ipynb                # SFT training loop
├── rl/                                 # Reinforcement Learning (DPO)
│   ├── download_ultrafeedback.ipynb    # Download UltraFeedback preference dataset
│   ├── tokenize_dpo.ipynb              # Tokenize DPO pairs
│   └── dpo_ultrafeedback.ipynb         # DPO training loop
└── test/                               # Evaluation & visualization
    ├── test_checkpoint.ipynb           # Eval + interactive chat
    └── visualize_model.ipynb           # Weight heatmaps + t-SNE embeddings
```

## Pipeline

```
data_* folders → tokenizer_pipeline → token_shards/ → pre_train → SFT → DPO → checkpoint
```

### 1. Acquire & Tokenize (Pre-train)

Download data into `data_*` folders on Google Drive:
- `download_c4.ipynb` — C4 English web text from HuggingFace
- `download_wikipedia.ipynb` — Full English Wikipedia
- `download_code.ipynb` — Code datasets

`tokenizer_pipeline.ipynb` scans all `data_*` folders and:
- Trains BPE tokenizer by sampling random files (500MB subset)
- Tokenizes each source file into an individual shard in `token_shards/`
- Tracks `tokenization_id` (hash of tokenizer.json)
- **Incremental:** only tokenizes new/changed files
- **Tokenizer change:** automatically wipes old shards and re-tokenizes everything

### 2. Pre-train

`pre_train_mar23.ipynb` reads shards directly:
- Shuffles shard order each epoch
- Loads one shard at a time into RAM (not memmap over Drive)
- Verifies `tokenization_id` matches checkpoint before resuming
- Saves checkpoint after each epoch

### 3. Supervised Fine-Tuning (SFT)

- `download_alpaca.ipynb` — Download Alpaca instruction-following dataset
- `tokenize_sft.ipynb` — Tokenize into instruction/response pairs
- `sft_alpaca.ipynb` — Fine-tune the pre-trained checkpoint on instruction data

### 4. Reinforcement Learning (DPO)

- `download_ultrafeedback.ipynb` — Download UltraFeedback preference dataset
- `tokenize_dpo.ipynb` — Tokenize chosen/rejected pairs
- `dpo_ultrafeedback.ipynb` — Direct Preference Optimization on the SFT checkpoint

### 5. Evaluate

- `test/test_checkpoint.ipynb` — Eval + interactive chat with any checkpoint
- `test/visualize_model.ipynb` — Weight heatmaps + t-SNE embeddings

## Data Versioning

Each pipeline step produces manifests in `sparkyllm/manifests/`:

- **tokenization** — tokenizer hash, shard count, total tokens, source file hashes
- **training** — checkpoint hash, full config, links to tokenization_id

The `shard_manifest.json` in `token_shards/` maps each shard to its source file with hashes, enabling incremental updates.

## Data on Google Drive

```
Google Drive/sparkyllm/
├── datasets_pretrain/
│   ├── data_books/
│   ├── data_c4/
│   ├── data_wikipedia/
│   └── tokenizer_out/
├── token_shards/        # Tokenized binary shards
│   ├── shard_0000.bin
│   ├── shard_0001.bin
│   ├── ...
│   ├── shard_manifest.json
│   └── meta.json
├── checkpoints/
└── manifests/
```

To add new data: create a `data_<name>/` folder with `.txt` files, re-run `tokenizer_pipeline` (only new files get tokenized), then `pre_train`.

## Hardware

Trained on NVIDIA A100-SXM4-80GB via Google Colab.
