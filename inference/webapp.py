"""SparkLLM Streamlit frontend — calls the inference API over HTTP."""
import streamlit as st
import requests
import os

# --- Configuration ---
DEFAULT_API_URL = "http://localhost:8000"
API_URL = os.environ.get("SPARKYLLM_API_URL", DEFAULT_API_URL)

st.set_page_config(page_title="SparkLLM", page_icon="⚡", layout="centered")

# --- Sidebar ---
st.sidebar.title("⚡ SparkLLM")
st.sidebar.markdown("A 650M parameter GPT trained from scratch on classic literature.")

api_url = st.sidebar.text_input("API URL", value=API_URL, help="The ngrok URL of your SparkLLM API")

st.sidebar.markdown("### Generation Settings")
max_tokens = st.sidebar.slider("Max tokens", 10, 500, 200)
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.8, step=0.05)
top_k = st.sidebar.slider("Top-K", 0, 200, 40)
top_p = st.sidebar.slider("Top-P", 0.1, 1.0, 0.9, step=0.05)

# --- Health check ---
health_ok = False
try:
    health = requests.get(f"{api_url}/health", timeout=5).json()
    health_ok = health.get("status") == "ok" and health.get("model_loaded")
    gpu_name = health.get("gpu", "CPU")
    st.sidebar.success(f"Connected — {gpu_name}")
except Exception:
    st.sidebar.error(f"Cannot reach API at {api_url}")

# --- Main area ---
st.title("⚡ SparkLLM")
st.caption("Type a prompt and the model will continue it.")

if "history" not in st.session_state:
    st.session_state.history = []

prompt = st.text_area("Enter your prompt:", height=100, placeholder="Once upon a time...")

col1, col2 = st.columns([1, 5])
generate_clicked = col1.button("Generate", type="primary", disabled=not health_ok)
if col2.button("Clear history"):
    st.session_state.history = []
    st.rerun()

if generate_clicked and prompt.strip():
    with st.spinner("Generating..."):
        try:
            resp = requests.post(
                f"{api_url}/generate",
                json={
                    "prompt": prompt.strip(),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            result = data["generated_text"]
            st.session_state.history.insert(0, {"prompt": prompt.strip(), "output": result})
        except requests.exceptions.RequestException as e:
            st.error(f"API error: {e}")

# --- Display history ---
for i, entry in enumerate(st.session_state.history):
    with st.container():
        st.markdown(f"**Prompt:** {entry['prompt']}")
        st.markdown(f"**Output:**")
        st.text_area(
            label=f"output_{i}",
            value=entry["output"],
            height=200,
            disabled=True,
            label_visibility="collapsed",
        )
        st.divider()
