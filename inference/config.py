"""SparkLLM inference configuration."""
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFERENCE_FILES = os.path.join(PROJECT_ROOT, "Inference_files")
CHECKPOINT_PATH = os.path.join(INFERENCE_FILES, "gpt_medium_phase2.pth")
TOKENIZER_PATH = os.path.join(INFERENCE_FILES, "tokenizer.json")
META_PATH = os.path.join(INFERENCE_FILES, "train_long_meta.json")

# Model architecture (must match training)
BLOCK_SIZE = 768
EMBED_DIM = 1280
NUM_LAYERS = 24
NUM_HEADS = EMBED_DIM // 64  # 20
FF_HIDDEN_DIM = 4 * EMBED_DIM  # 5120

# Server
HOST = "0.0.0.0"
PORT = 8000
