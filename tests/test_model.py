"""Tests for SparkLLM model loading and generation."""
import pytest
import torch
from inference.config import CHECKPOINT_PATH, TOKENIZER_PATH, META_PATH, BLOCK_SIZE
from inference.model import SimpleGPT, get_vocab_size, load_model, load_tokenizer, generate
import os


# --- Config & file tests ---

def test_inference_files_exist():
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
    assert os.path.exists(TOKENIZER_PATH), f"Tokenizer not found: {TOKENIZER_PATH}"
    assert os.path.exists(META_PATH), f"Metadata not found: {META_PATH}"


def test_vocab_size():
    vocab_size = get_vocab_size()
    assert vocab_size > 0
    assert vocab_size % 64 == 0, "Vocab size should be padded to multiple of 64"


# --- Model architecture tests ---

def test_model_architecture():
    vocab_size = get_vocab_size()
    model = SimpleGPT(vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 100_000_000, f"Expected >100M params, got {total_params:,}"
    assert len(model.blocks) == 24


def test_model_forward_shape():
    vocab_size = get_vocab_size()
    model = SimpleGPT(vocab_size)
    model.eval()
    x = torch.randint(0, vocab_size, (1, 16))
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 16, vocab_size)


# --- Loading tests (require GPU + checkpoint) ---

@pytest.fixture(scope="module")
def loaded_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_model(device)


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


def test_load_model(loaded_model):
    assert loaded_model is not None
    assert not loaded_model.training, "Model should be in eval mode"


def test_load_tokenizer(tokenizer):
    assert tokenizer is not None
    encoded = tokenizer.encode("Hello world")
    assert len(encoded.ids) > 0


def test_tokenizer_special_tokens(tokenizer):
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    assert eot_id is not None and eot_id == 0


# --- Generation tests ---

def test_generate_produces_text(loaded_model, tokenizer):
    output = generate(loaded_model, tokenizer, "The king", max_tokens=20, temperature=0.8)
    assert isinstance(output, str)
    assert len(output) > len("The king")


def test_generate_respects_max_tokens(loaded_model, tokenizer):
    output = generate(loaded_model, tokenizer, "Once upon", max_tokens=5, temperature=0.8)
    ids = tokenizer.encode(output).ids
    prompt_ids = tokenizer.encode("Once upon").ids
    # Generated at most 5 new tokens (could be fewer if EOT hit)
    assert len(ids) <= len(prompt_ids) + 5


def test_generate_deterministic_with_low_temp(loaded_model, tokenizer):
    """Very low temperature should produce near-deterministic output."""
    out1 = generate(loaded_model, tokenizer, "In the beginning", max_tokens=10, temperature=0.01, top_k=1)
    out2 = generate(loaded_model, tokenizer, "In the beginning", max_tokens=10, temperature=0.01, top_k=1)
    assert out1 == out2


def test_generate_long_prompt_truncates(loaded_model, tokenizer):
    """Prompts longer than BLOCK_SIZE should not crash (truncated to last BLOCK_SIZE tokens)."""
    long_prompt = "word " * (BLOCK_SIZE + 100)
    output = generate(loaded_model, tokenizer, long_prompt, max_tokens=5, temperature=0.8)
    assert isinstance(output, str)
