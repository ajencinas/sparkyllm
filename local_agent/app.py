"""Streamlit interface for the local sparkyllm agent.

Run with: streamlit run app.py
"""
from __future__ import annotations

import os
import sys

import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_TEST = os.path.normpath(os.path.join(_HERE, "..", "local_test"))
sys.path.insert(0, _LOCAL_TEST)
sys.path.insert(0, _HERE)

from sparky_model import (  # type: ignore
    detect_device,
    load_model,
    load_tokenizer,
    vocab_size_for,
)
from agent import AgentResult, AgentRunner

WEIGHTS_DIR = os.path.join(_LOCAL_TEST, "weights")
TOKENIZER_PATH = os.path.join(WEIGHTS_DIR, "tokenizer.json")

STAGE_FILES = {
    "dpo": "gpt_medium_dpo_best.pth",
    "tools": "gpt_medium_tools_best.pth",
    "sft": "gpt_medium_sft_best.pth",
    "pretrain": "gpt_medium_phase2.pth",
}


def checkpoint_path(stage: str) -> str:
    return os.path.join(WEIGHTS_DIR, STAGE_FILES[stage])


# ---- Cached loaders ----

@st.cache_resource(show_spinner="Loading tokenizer…")
def get_tokenizer():
    return load_tokenizer(TOKENIZER_PATH)


@st.cache_resource(show_spinner="Loading model…", max_entries=1)
def get_runner(stage: str, max_steps: int, temperature: float):
    device = detect_device()
    tokenizer = get_tokenizer()
    vocab = vocab_size_for(tokenizer)
    model = load_model(checkpoint_path(stage), vocab, device)
    runner = AgentRunner(
        model, tokenizer, device,
        max_steps=max_steps,
        temperature=temperature,
    )
    return runner


# ---- App ----

st.set_page_config(page_title="SparkyLLM Agent", page_icon="🤖", layout="centered")
st.title("🤖 SparkyLLM local agent")

if not os.path.exists(TOKENIZER_PATH):
    st.error(f"Tokenizer not found at `{TOKENIZER_PATH}`")
    st.caption(f"Drop `tokenizer.json` into `{WEIGHTS_DIR}/` first.")
    st.stop()

with st.sidebar:
    st.header("Model")
    stage = st.selectbox("Stage", list(STAGE_FILES.keys()), index=0)
    if not os.path.exists(checkpoint_path(stage)):
        st.error(f"Missing weights file: `{STAGE_FILES[stage]}`")
        st.caption(f"Drop it in `{WEIGHTS_DIR}/`")
        st.stop()

    st.header("Agent")
    max_steps = st.slider("Max tool calls per turn", 1, 5, 3)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, step=0.05)

    st.divider()
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

try:
    runner = get_runner(stage, max_steps, temperature)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.sidebar.success(f"Loaded `{stage}` on **{runner.device.type.upper()}**")

if "history" not in st.session_state:
    st.session_state.history = []


def render_trace(result: AgentResult) -> None:
    n = len(result.steps)
    label = f"agent trace ({n} step{'s' if n != 1 else ''}{', truncated' if result.truncated else ''})"
    with st.expander(label):
        if not result.steps:
            st.caption("No tool calls. Model answered directly.")
        for i, step in enumerate(result.steps, 1):
            st.markdown(f"**Step {i}**")
            if step.thought:
                st.markdown(f"_Thought:_ {step.thought}")
            st.markdown(f"_Action:_ `{step.action}`")
            if step.input:
                st.markdown(f"_Input:_ `{step.input}`")
            if step.error:
                st.error(f"Error: {step.error}")
            else:
                st.markdown(f"_Result:_ `{step.result}`")
        if result.raw_trace:
            st.divider()
            st.caption("Raw trace:")
            st.code(result.raw_trace, language="text")


# Render history
for turn in st.session_state.history:
    if turn["role"] == "user":
        with st.chat_message("user"):
            st.markdown(turn["content"])
    else:
        with st.chat_message("assistant"):
            result: AgentResult = turn["result"]
            st.markdown(result.final_answer or "_(no final answer)_")
            render_trace(result)

# Input
user_input = st.chat_input("Ask the agent…")
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("agent thinking…"):
            result = runner.run_turn(user_input)
        st.markdown(result.final_answer or "_(no final answer)_")
        render_trace(result)

    st.session_state.history.append({"role": "assistant", "result": result})
