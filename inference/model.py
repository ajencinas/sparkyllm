"""SparkLLM model definition and loading utilities."""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from .config import (
    BLOCK_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, FF_HIDDEN_DIM,
    CHECKPOINT_PATH, TOKENIZER_PATH, META_PATH,
)


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
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
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim * 2)
        self.w2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x1, x2 = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
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
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.blocks = nn.ModuleList(
            [TransformerBlock(EMBED_DIM, NUM_HEADS, FF_HIDDEN_DIM) for _ in range(NUM_LAYERS)]
        )
        self.final_norm = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(vocab_size, EMBED_DIM, bias=False)  # placeholder, tied below
        self.lm_head = nn.Linear(EMBED_DIM, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


def get_vocab_size() -> int:
    """Read vocab size from metadata, pad to multiple of 64 for tensor cores."""
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    vocab_size = int(meta["vocab_size"])
    if vocab_size % 64 != 0:
        vocab_size = ((vocab_size + 63) // 64) * 64
    return vocab_size


def load_model(device: torch.device | None = None) -> SimpleGPT:
    """Load the pretrained SimpleGPT model onto the given device."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = get_vocab_size()
    model = SimpleGPT(vocab_size).to(device)

    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    # Strip _orig_mod. prefix from torch.compile'd checkpoints
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model_state = model.state_dict()
    matched = {k: v for k, v in clean_state.items()
               if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(matched, strict=False)
    model.eval()

    print(f"Loaded {len(matched)}/{len(model_state)} layers on {device}")
    return model


def load_tokenizer() -> Tokenizer:
    """Load the BPE tokenizer."""
    return Tokenizer.from_file(TOKENIZER_PATH)


@torch.no_grad()
def _apply_repetition_penalty(logits, token_ids, penalty):
    """Penalize tokens that already appeared. penalty=1.0 means no effect."""
    if penalty == 1.0 or len(token_ids) == 0:
        return logits
    seen = torch.tensor(list(set(token_ids)), dtype=torch.long, device=logits.device)
    scores = logits[:, seen]
    # Divide positive logits, multiply negative logits (standard HF approach)
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    logits[:, seen] = scores
    return logits


def generate(
    model: SimpleGPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    device: torch.device | None = None,
) -> str:
    """Generate text continuation from a prompt."""
    if device is None:
        device = next(model.parameters()).device

    eot_id = tokenizer.token_to_id("<|endoftext|>")
    ids = tokenizer.encode(prompt).ids
    tokens = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        context = tokens[:, -BLOCK_SIZE:]
        logits = model(context)[:, -1, :]
        logits = _apply_repetition_penalty(logits, tokens[0].tolist(), repetition_penalty)
        logits = logits / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float('-inf')
            logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == eot_id:
            break

        tokens = torch.cat([tokens, next_token], dim=1)

    return tokenizer.decode(tokens[0].tolist())


@torch.no_grad()
def generate_stream(
    model: SimpleGPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    device: torch.device | None = None,
):
    """Yield token strings one at a time as they're generated."""
    if device is None:
        device = next(model.parameters()).device

    eot_id = tokenizer.token_to_id("<|endoftext|>")
    ids = tokenizer.encode(prompt).ids
    tokens = torch.tensor([ids], dtype=torch.long, device=device)

    # Yield the prompt first
    yield tokenizer.decode(ids)

    for _ in range(max_tokens):
        context = tokens[:, -BLOCK_SIZE:]
        logits = model(context)[:, -1, :]
        logits = _apply_repetition_penalty(logits, tokens[0].tolist(), repetition_penalty)
        logits = logits / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float('-inf')
            logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == eot_id:
            break

        tokens = torch.cat([tokens, next_token], dim=1)
        yield tokenizer.decode([next_token.item()])
