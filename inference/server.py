"""SparkLLM FastAPI inference server with ngrok tunnel."""
import argparse
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .model import load_model, load_tokenizer, generate as model_generate
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


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    device: str


# --- Endpoints ---

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
        device=_device,
    )
    return GenerateResponse(
        prompt=req.prompt,
        generated_text=text,
        device=str(_device),
    )


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
