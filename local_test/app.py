"""Streamlit local test for sparkyllm. Run with: streamlit run app.py"""
from __future__ import annotations

import os
import sys

import streamlit as st

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparky_model import (
    BLOCK_SIZE,
    detect_device,
    load_model,
    load_tokenizer,
    stream_generate,
    vocab_size_for,
)

# ---- Paths ----
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
TOKENIZER_PATH = os.path.join(WEIGHTS_DIR, "tokenizer.json")

STAGE_FILES = {
    "dpo": "gpt_medium_dpo_best.pth",
    "sft": "gpt_medium_sft_best.pth",
    "pretrain": "gpt_medium_phase2.pth",
}


def checkpoint_path(stage: str) -> str:
    return os.path.join(WEIGHTS_DIR, STAGE_FILES[stage])


# ---- Cached loaders ----

@st.cache_resource(show_spinner="Loading tokenizer…")
def get_tokenizer():
    return load_tokenizer(TOKENIZER_PATH)


@st.cache_resource(show_spinner="Loading model (slow on CPU)…", max_entries=1)
def get_model(stage: str):
    device = detect_device()
    tokenizer = get_tokenizer()
    vocab = vocab_size_for(tokenizer)
    model = load_model(checkpoint_path(stage), vocab, device)
    return model, device


# ---- App ----

st.set_page_config(page_title="SparkyLLM", page_icon="⚡", layout="centered")
st.title("⚡ SparkyLLM local test")

if not os.path.exists(TOKENIZER_PATH):
    st.error(f"Tokenizer not found at `{TOKENIZER_PATH}`")
    st.caption("Download `tokenizer.json` from "
               "`MyDrive/sparkyllm/datasets_pretrain/tokenizer_out/` and place it in `local_test/weights/`.")
    st.stop()

# ---- Sidebar ----
with st.sidebar:
    st.header("Model")
    stage = st.selectbox(
        "Checkpoint stage", list(STAGE_FILES.keys()), index=0,
        help="Which trained checkpoint to load. The file must exist under local_test/weights/.",
    )
    if not os.path.exists(checkpoint_path(stage)):
        st.error(f"Missing weights file: `{STAGE_FILES[stage]}`")
        st.caption(f"Drop it in `{WEIGHTS_DIR}/`")
        st.stop()

    mode = st.radio(
        "Mode",
        ["Chat (Instruction/Response)", "Continue (raw text)"],
        index=0,
        help="Chat wraps each input as Alpaca-format instruction. "
             "Continue appends raw text and lets the model continue.",
    )
    chat_mode = mode.startswith("Chat")

    use_history = st.checkbox(
        "Feed prior turns to model",
        value=(not chat_mode),
        help="If on, prior turns are included in the model context. "
             "Off by default for Chat mode (SFT/DPO models are single-turn).",
    )

    st.header("Generation")
    max_tokens = st.slider("Max new tokens", 10, 500, 200, step=10)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, step=0.05)
    top_k = st.slider("Top-k", 0, 200, 40)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.9, step=0.05)
    rep_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.1, step=0.05)

    st.divider()
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()


# ---- Load model ----
try:
    model, device = get_model(stage)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

tokenizer = get_tokenizer()
eot_id = tokenizer.token_to_id("<|endoftext|>")

st.sidebar.success(f"Loaded `{stage}` on **{device.type.upper()}**")

# ---- History state ----
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": ..., "content": ...}

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# ---- Input ----
user_input = st.chat_input("Type a prompt…")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build the prompt the model actually sees
    if chat_mode:
        current = f"Instruction: {user_input}\nResponse: "
    else:
        current = user_input

    if use_history and len(st.session_state.history) > 1:
        prior_parts = []
        for turn in st.session_state.history[:-1]:  # exclude the just-appended input
            if chat_mode:
                if turn["role"] == "user":
                    prior_parts.append(f"Instruction: {turn['content']}")
                else:
                    prior_parts.append(f"Response: {turn['content']}")
            else:
                prior_parts.append(turn["content"])
        full_prompt = "\n".join(prior_parts) + "\n" + current
    else:
        full_prompt = current

    prompt_ids = tokenizer.encode(full_prompt).ids
    # Trim from the left so prompt + max_tokens fits in BLOCK_SIZE
    budget = BLOCK_SIZE - max_tokens
    if budget > 0 and len(prompt_ids) > budget:
        prompt_ids = prompt_ids[-budget:]

    with st.chat_message("assistant"):
        gen = stream_generate(
            model, tokenizer, prompt_ids,
            device=device,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=rep_penalty,
            eot_id=eot_id,
        )
        response = st.write_stream(gen)

    st.session_state.history.append({"role": "assistant", "content": response})
