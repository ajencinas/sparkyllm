"""SparkyLLM model definition + load + streaming generate (CPU/MPS/CUDA)."""
from __future__ import annotations

import os
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# ---- Architecture (must match training, see pretrain/pre_train_mar23.ipynb) ----
BLOCK_SIZE = 768
EMBED_DIM = 1280
NUM_LAYERS = 24
NUM_HEADS = EMBED_DIM // 64  # 20
FF_HIDDEN_DIM = 4 * EMBED_DIM  # 5120


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim * 2)
        self.w2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x1, x2 = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = SwiGLU(embed_dim, ff_hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class SimpleGPT(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.blocks = nn.ModuleList(
            [TransformerBlock(EMBED_DIM, NUM_HEADS, FF_HIDDEN_DIM) for _ in range(NUM_LAYERS)]
        )
        self.final_norm = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


# ---- Helpers ----

def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    return Tokenizer.from_file(tokenizer_path)


def vocab_size_for(tokenizer: Tokenizer) -> int:
    """Tokenizer's raw vocab size, padded to a multiple of 64 (matches training)."""
    v = tokenizer.get_vocab_size()
    if v % 64 != 0:
        v = ((v + 63) // 64) * 64
    return v


def load_model(checkpoint_path: str, vocab_size: int, device: torch.device) -> SimpleGPT:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = SimpleGPT(vocab_size)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Strip torch.compile prefix if present
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    msd = model.state_dict()
    safe = {k: v for k, v in state.items() if k in msd and v.shape == msd[k].shape}
    model.load_state_dict(safe, strict=False)
    missing = len(msd) - len(safe)
    if missing:
        print(f"warning: {missing}/{len(msd)} model keys not loaded from checkpoint")

    if device.type == "cuda":
        model = model.to(device, dtype=torch.bfloat16)
    else:
        model = model.to(device, dtype=torch.float32)
    model.eval()
    return model


@torch.no_grad()
def stream_generate(
    model: SimpleGPT,
    tokenizer: Tokenizer,
    prompt_ids: list[int],
    *,
    device: torch.device,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    rep_window: int = 256,
    eot_id: Optional[int] = None,
    stop_strings: Optional[list[str]] = None,
) -> Iterator[str]:
    """Yield decoded text suffix per token. Handles multi-byte UTF-8 correctly.

    If `stop_strings` is given, generation halts as soon as any of those strings
    appears in the generated-so-far text. The matched stop string IS yielded as
    part of the output (caller is responsible for stripping it).
    """
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    new_ids: list[int] = []
    prev_text = tokenizer.decode(prompt_ids)
    generated_so_far = ""

    for _ in range(max_new_tokens):
        context = tokens[:, -BLOCK_SIZE:]
        logits = model(context)[:, -1, :].float()

        # Repetition penalty (windowed)
        if repetition_penalty > 1.0:
            recent = set(tokens[0, -rep_window:].tolist())
            for tid in recent:
                if tid < logits.size(-1):
                    if logits[0, tid] > 0:
                        logits[0, tid] /= repetition_penalty
                    else:
                        logits[0, tid] *= repetition_penalty

        logits = logits / max(temperature, 1e-6)

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = (cum - sorted_probs) >= top_p
            sorted_logits[mask] = float("-inf")
            logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        tid = next_id.item()

        if eot_id is not None and tid == eot_id:
            return

        new_ids.append(tid)
        tokens = torch.cat([tokens, next_id], dim=1)

        # Incremental decode: emit only the new suffix
        full_text = tokenizer.decode(prompt_ids + new_ids)
        new_piece = full_text[len(prev_text):]
        if new_piece:
            yield new_piece
            prev_text = full_text
            generated_so_far += new_piece
            if stop_strings:
                if any(s in generated_so_far for s in stop_strings):
                    return
