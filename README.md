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
├── pretrain/                    # Training pipeline (Colab notebooks)
│   ├── download_c4.ipynb        # Download C4 dataset from HuggingFace
│   ├── consolidate_data.ipynb   # Merge all data_* folders into one text file
│   ├── tokenizer_pipeline.ipynb # Train tokenizer + encode text to token binary
│   ├── pre_train_mar23.ipynb    # Main training loop
│   ├── test_checkpoint.ipynb    # Eval: perplexity, repetition, diversity, interactive chat
│   ├── visualize_model.ipynb    # Weight heatmaps + t-SNE embedding visualization
│   ├── bootstrap_manifest.ipynb # One-time manifest creation from existing files
│   └── explore_c4.py           # Local script to preview C4 dataset samples
├── data_processing/             # Data cleaning utilities
│   └── clean_gutemberg.py       # Gutenberg text cleaner
```

## Pipeline

Run in order on Google Colab:

1. **download_c4.ipynb** — Stream documents from C4 English into `data_c4/`
2. **consolidate_data.ipynb** — Scan all `data_*` folders, merge into `training_data_long.txt`
3. **tokenizer_pipeline.ipynb** — Train BPE tokenizer (or load existing), encode to `train_long.bin`
4. **pre_train_mar23.ipynb** — Train the model on A100, saves checkpoints to Google Drive

## Data Versioning

Each pipeline step produces a JSON manifest in `sparkyllm/manifests/` on Google Drive:

- **consolidation** — hashes of every source file across all `data_*` folders
- **tokenization** — tokenizer hash (`tokenization_id`), token counts, vocab config
- **training** — checkpoint hash, full training config, links back to tokenization_id

The training notebook automatically checks that the checkpoint's `tokenization_id` matches the current tokenizer before resuming. If the tokenizer changed, it trains from scratch instead of loading incompatible weights.

## Data

Training data is stored on Google Drive (not in this repo):

```
Google Drive/sparkyllm/
├── datasets_pretrain/
│   ├── data_books/          # Classic literature (Project Gutenberg)
│   ├── data_c4/             # C4 English web text
│   └── tokenizer_out/       # tokenizer.json + train_long.bin
├── checkpoints/             # Model weights
└── manifests/               # Data versioning JSONs
```

To add new data sources, create a `data_<name>/` folder with `.txt` files and re-run the pipeline from step 2.

## Hardware

Trained on NVIDIA A100-SXM4-80GB via Google Colab.
