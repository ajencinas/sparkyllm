"""SparkLLM FastAPI inference server with ngrok tunnel and built-in web UI."""
import argparse
import json
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field
from .model import load_model, load_tokenizer, generate as model_generate, generate_stream
from .config import HOST, PORT

# --- Globals filled at startup ---
_model = None
_tokenizer = None
_device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {_device}...")
    _model = load_model(_device)
    _tokenizer = load_tokenizer()
    print("Model ready.")
    yield
    print("Shutting down.")


app = FastAPI(title="SparkLLM", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response schemas ---

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    max_tokens: int = Field(200, ge=1, le=1000)
    temperature: float = Field(0.8, ge=0.01, le=2.0)
    top_k: int = Field(40, ge=0, le=500)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0)


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    device: str


# --- API Endpoints ---

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "device": str(_device) if _device else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate_text(req: GenerateRequest):
    text = model_generate(
        model=_model,
        tokenizer=_tokenizer,
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        device=_device,
    )
    return GenerateResponse(
        prompt=req.prompt,
        generated_text=text,
        device=str(_device),
    )


@app.post("/generate/stream")
def generate_text_stream(req: GenerateRequest):
    def event_stream():
        for token in generate_stream(
            model=_model,
            tokenizer=_tokenizer,
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            device=_device,
        ):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True, 'device': str(_device)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- Web UI (served at /) ---

WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SparkLLM</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0a0a0f; color: #e0e0e0; min-height: 100vh; }
  .container { max-width: 800px; margin: 0 auto; padding: 2rem 1.5rem; }
  h1 { font-size: 2rem; margin-bottom: 0.25rem; }
  h1 span { color: #fbbf24; }
  .subtitle { color: #888; margin-bottom: 2rem; font-size: 0.95rem; }
  .status { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 1rem;
            font-size: 0.8rem; margin-bottom: 1.5rem; }
  .status.ok { background: #065f46; color: #6ee7b7; }
  .status.err { background: #7f1d1d; color: #fca5a5; }
  textarea { width: 100%; padding: 1rem; border-radius: 0.5rem; border: 1px solid #333;
             background: #1a1a2e; color: #e0e0e0; font-size: 1rem; resize: vertical;
             font-family: inherit; }
  textarea:focus { outline: none; border-color: #fbbf24; }
  .controls { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; align-items: end; }
  .control-group { display: flex; flex-direction: column; gap: 0.25rem; }
  .control-group label { font-size: 0.75rem; color: #888; }
  .control-group input { width: 80px; padding: 0.4rem; border-radius: 0.3rem;
                         border: 1px solid #333; background: #1a1a2e; color: #e0e0e0;
                         font-size: 0.9rem; }
  button { padding: 0.6rem 2rem; border-radius: 0.5rem; border: none; cursor: pointer;
           font-size: 1rem; font-weight: 600; transition: all 0.15s; }
  #generate-btn { background: #fbbf24; color: #000; }
  #generate-btn:hover { background: #f59e0b; }
  #generate-btn:disabled { background: #555; color: #999; cursor: not-allowed; }
  .spinner { display: inline-block; width: 1rem; height: 1rem; border: 2px solid #999;
             border-top-color: #fbbf24; border-radius: 50%; animation: spin 0.6s linear infinite;
             margin-left: 0.5rem; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .output { margin-top: 2rem; }
  .output-card { background: #1a1a2e; border: 1px solid #333; border-radius: 0.5rem;
                 padding: 1.25rem; margin-bottom: 1rem; }
  .output-card .prompt-label { font-size: 0.75rem; color: #fbbf24; margin-bottom: 0.5rem;
                               text-transform: uppercase; letter-spacing: 0.05em; }
  .output-card .text { white-space: pre-wrap; line-height: 1.6; }
  .output-card .meta { font-size: 0.75rem; color: #666; margin-top: 0.75rem; }
  .listen-btn { background: none; border: 1px solid #555; color: #ccc; padding: 0.3rem 0.8rem;
                border-radius: 0.3rem; font-size: 0.8rem; cursor: pointer; margin-top: 0.5rem; }
  .listen-btn:hover { border-color: #fbbf24; color: #fbbf24; }
  .listen-btn.speaking { border-color: #f87171; color: #f87171; }
</style>
</head>
<body>
<div class="container">
  <h1><span>&#9889;</span> SparkLLM</h1>
  <p class="subtitle">650M parameter GPT trained from scratch on classic literature</p>
  <div id="status" class="status">Checking...</div>

  <textarea id="prompt" rows="4" placeholder="Once upon a time..."></textarea>

  <div class="controls">
    <div class="control-group">
      <label>Max Tokens</label>
      <input type="number" id="max_tokens" value="200" min="10" max="500">
    </div>
    <div class="control-group">
      <label>Temperature</label>
      <input type="number" id="temperature" value="0.8" min="0.1" max="2.0" step="0.05">
    </div>
    <div class="control-group">
      <label>Top-K</label>
      <input type="number" id="top_k" value="40" min="0" max="200">
    </div>
    <div class="control-group">
      <label>Top-P</label>
      <input type="number" id="top_p" value="0.9" min="0.1" max="1.0" step="0.05">
    </div>
    <div class="control-group">
      <label>Rep. Penalty</label>
      <input type="number" id="repetition_penalty" value="1.2" min="1.0" max="2.0" step="0.05">
    </div>
    <button id="generate-btn" disabled>Generate</button>
  </div>

  <div id="output" class="output"></div>
</div>

<script>
const statusEl = document.getElementById('status');
const btn = document.getElementById('generate-btn');
const promptEl = document.getElementById('prompt');
const outputEl = document.getElementById('output');

async function checkHealth() {
  try {
    const r = await fetch('/health', {headers: {'ngrok-skip-browser-warning': 'true'}});
    const d = await r.json();
    if (d.status === 'ok' && d.model_loaded) {
      statusEl.textContent = 'Connected — ' + (d.gpu || 'CPU');
      statusEl.className = 'status ok';
      btn.disabled = false;
    } else {
      throw new Error('Model not loaded');
    }
  } catch(e) {
    statusEl.textContent = 'API unavailable';
    statusEl.className = 'status err';
    btn.disabled = true;
  }
}

async function generate() {
  const prompt = promptEl.value.trim();
  if (!prompt) return;

  btn.disabled = true;
  btn.innerHTML = 'Generating<span class="spinner"></span>';

  const cardId = 'card-' + Date.now();
  const card = document.createElement('div');
  card.className = 'output-card';
  card.innerHTML = '<div class="prompt-label">Prompt: ' + escHtml(prompt) + '</div>'
    + '<div class="text" id="' + cardId + '"></div>'
    + '<button class="listen-btn" style="display:none">&#9654; Listen</button>'
    + '<div class="meta" id="meta-' + cardId + '"></div>';
  outputEl.prepend(card);

  const textEl = document.getElementById(cardId);
  const listenBtn = card.querySelector('.listen-btn');
  listenBtn.addEventListener('click', function() { toggleSpeech(this, cardId); });
  const metaEl = document.getElementById('meta-' + cardId);

  try {
    const r = await fetch('/generate/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true'},
      body: JSON.stringify({
        prompt,
        max_tokens: parseInt(document.getElementById('max_tokens').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        top_k: parseInt(document.getElementById('top_k').value),
        top_p: parseFloat(document.getElementById('top_p').value),
        repetition_penalty: parseFloat(document.getElementById('repetition_penalty').value),
      })
    });

    if (!r.ok) {
      const d = await r.json();
      throw new Error(d.detail || 'API error');
    }

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});

      const lines = buffer.split(String.fromCharCode(10));
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const data = JSON.parse(line.slice(6));
          if (data.done) {
            metaEl.textContent = 'Device: ' + data.device;
            listenBtn.style.display = '';
          } else if (data.token !== undefined) {
            textEl.textContent += data.token;
          }
        } catch(ignored) {}
      }
    }
  } catch(e) {
    textEl.textContent += ' [Error: ' + e.message + ']';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate';
  }
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function toggleSpeech(btn, textId) {
  if (speechSynthesis.speaking) {
    speechSynthesis.cancel();
    document.querySelectorAll('.listen-btn').forEach(b => {
      b.classList.remove('speaking');
      b.innerHTML = '&#9654; Listen';
    });
    return;
  }
  const text = document.getElementById(textId).textContent;
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 1.0;
  utter.pitch = 1.0;
  btn.classList.add('speaking');
  btn.innerHTML = '&#9632; Stop';
  utter.onend = () => {
    btn.classList.remove('speaking');
    btn.innerHTML = '&#9654; Listen';
  };
  speechSynthesis.speak(utter);
}

btn.addEventListener('click', generate);
promptEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) generate();
});

checkHealth();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def web_ui():
    return WEB_UI_HTML


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="SparkLLM inference server")
    parser.add_argument("--ngrok", action="store_true", help="Expose via ngrok tunnel")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    if args.ngrok:
        from pyngrok import ngrok
        public_url = ngrok.connect(args.port, "http")
        print(f"\n{'='*60}")
        print(f"  NGROK PUBLIC URL: {public_url}")
        print(f"{'='*60}\n")

    uvicorn.run(app, host=HOST, port=args.port)


if __name__ == "__main__":
    main()
