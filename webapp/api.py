"""
FastAPI Inference Server with Redis Caching
============================================
Replaces the basic Flask server with a production-grade async API.

Endpoints
---------
  GET  /              → Web UI (index.html)
  GET  /health        → model + cache status
  GET  /metrics       → session stats (requests, cache hits, latency)
  POST /autocomplete  → { prompt } → { predictions, cached, latency_ms }

Features
--------
  - Async request handling via FastAPI + Uvicorn
  - Redis cache: same prompt → instant response, no model inference
  - Cache key = SHA256(prompt) — collision-proof
  - LRU eviction with 1-hour TTL
  - Pydantic request/response validation
  - /metrics endpoint for dashboard

Run
---
  uvicorn webapp.api:app --host 0.0.0.0 --port 5001 --reload
  OR
  python webapp/api.py
"""

import asyncio
import hashlib
import json
import os
import sys
import threading
import time
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

# ── package path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import redis.asyncio as aioredis

from python_autocomplete.dataset.python_tokenizer import PythonTokenizer
from python_autocomplete.dataset.ast_features import ASTFeatureExtractor
from python_autocomplete.models.syntax_transformer import SyntaxAwareTransformer

# ── same fallback training corpus as flask server ─────────────────
TRAINING_CODE = '''
import torch
import torch.nn as nn
import torch.optim as optim

class LinearModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Trainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(self, data, labels):
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, data, labels):
        self.model.eval()
        with torch.no_grad():
            output = self.model(data)
            pred = output.argmax(dim=-1)
            accuracy = (pred == labels).float().mean()
        return accuracy.item()

def train_model(model, data, labels, epochs=10):
    trainer = Trainer(model)
    for epoch in range(epochs):
        loss = trainer.train_step(data, labels)
    return model

def predict(model, x):
    model.eval()
    with torch.no_grad():
        output = model(x)
        return output.argmax(dim=-1)

class DataLoader:
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = torch.randperm(len(self.data)) if self.shuffle else torch.arange(len(self.data))
        for i in range(0, len(indices), self.batch_size):
            yield self.data[indices[i:i + self.batch_size]]
'''

# ── global model state ────────────────────────────────────────────
_model:     Optional[nn.Module] = None
_tokenizer: Optional[PythonTokenizer] = None
_model_lock = threading.Lock()
_ready      = False
_status     = "Initializing…"
_model_info = {}

# ── session metrics ───────────────────────────────────────────────
_metrics = {
    "total_requests":   0,
    "cache_hits":       0,
    "cache_misses":     0,
    "errors":           0,
    "total_latency_ms": 0.0,
}

# ── Redis config ──────────────────────────────────────────────────
REDIS_URL   = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL   = int(os.getenv("CACHE_TTL", "3600"))   # 1 hour default
_redis: Optional[aioredis.Redis] = None


# ═══════════════════════════════════════════════════════════════════
# Model loading (same logic as flask server)
# ═══════════════════════════════════════════════════════════════════

def _load_model():
    global _model, _tokenizer, _ready, _status, _model_info

    try:
        bundle_path = os.path.join(os.path.dirname(__file__), 'model_bundle.pt')

        if os.path.exists(bundle_path):
            _status = "Loading pre-trained model…"
            bundle  = torch.load(bundle_path, map_location='cpu', weights_only=False)
            tok     = bundle['tokenizer']

            from labml_nn.transformers import TransformerConfigs
            D = bundle.get('d_model',  512)
            L = bundle.get('n_layers', 6)

            t_conf = TransformerConfigs()
            t_conf.d_model     = D
            t_conf.n_layers    = L
            t_conf.n_src_vocab = tok.n_tokens
            t_conf.n_tgt_vocab = tok.n_tokens
            t_conf.dropout     = 0.0

            model = SyntaxAwareTransformer(
                n_tokens=tok.n_tokens, d_model=D,
                encoder=t_conf.encoder, dropout=0.0, use_copy=True)
            model.load_state_dict(bundle['model_state'])
            model.eval()

            _tokenizer = tok
            _model     = model
            _model_info = {
                "type":   "pre-trained",
                "vocab":  tok.n_tokens,
                "params": sum(p.numel() for p in model.parameters()),
                "epoch":  bundle.get('epoch', '?'),
                "d_model": D,
                "n_layers": L,
            }
            _ready  = True
            _status = (f"Ready (pre-trained) | vocab={tok.n_tokens} "
                       f"params={_model_info['params']:,} epoch={_model_info['epoch']}")
            return

        # ── fallback: train demo model ────────────────────────────
        _status = "Building tokenizer…"
        tok = PythonTokenizer()
        tok.train(TRAINING_CODE)

        _status = "Building model…"
        from labml_nn.transformers import TransformerConfigs
        t_conf = TransformerConfigs()
        t_conf.d_model     = 256
        t_conf.n_layers    = 4
        t_conf.n_src_vocab = tok.n_tokens
        t_conf.n_tgt_vocab = tok.n_tokens
        t_conf.dropout     = 0.1

        model = SyntaxAwareTransformer(
            n_tokens=tok.n_tokens, d_model=256,
            encoder=t_conf.encoder, dropout=0.1, use_copy=True)

        _status = "Training demo model (~30s)…"
        ids, type_ids = tok.encode_with_types(TRAINING_CODE)
        extractor = ASTFeatureExtractor()
        feats = extractor.extract(TRAINING_CODE)
        feats.pad_or_trim(len(ids))

        mk  = lambda lst: torch.tensor(lst, dtype=torch.long).unsqueeze(1)
        src = mk(ids)

        opt  = optim.Adam(model.parameters(), lr=5e-4)
        loss_fn = nn.CrossEntropyLoss()
        model.train()

        for epoch in range(1, 81):
            opt.zero_grad()
            logits, _ = model(src[:-1], None)
            loss = loss_fn(logits[:, 0, :], src[1:, 0])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if epoch % 20 == 0:
                _status = f"Training… {epoch}/80  loss={loss.item():.3f}"

        model.eval()
        _tokenizer = tok
        _model     = model
        _model_info = {
            "type":   "demo",
            "vocab":  tok.n_tokens,
            "params": sum(p.numel() for p in model.parameters()),
        }
        _ready  = True
        _status = f"Ready (demo) | vocab={tok.n_tokens}"

    except Exception as e:
        _status = f"Error: {e}"
        raise


threading.Thread(target=_load_model, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════
# FastAPI app
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Python Autocomplete API",
    description="Syntax-aware Python code autocomplete with Redis caching",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────

class AutocompleteRequest(BaseModel):
    prompt: str = Field(..., description="Code typed so far")
    top_k:  int = Field(8, ge=1, le=20, description="Max suggestions to return")


class Prediction(BaseModel):
    text: str
    prob: float


class AutocompleteResponse(BaseModel):
    success:     bool
    predictions: List[Prediction] = []
    cached:      bool = False
    latency_ms:  float = 0.0
    reason:      Optional[str] = None


class HealthResponse(BaseModel):
    ready:       bool
    status:      str
    model_info:  dict = {}
    cache_online: bool = False


class MetricsResponse(BaseModel):
    total_requests:   int
    cache_hits:       int
    cache_misses:     int
    cache_hit_rate:   str
    errors:           int
    avg_latency_ms:   str


# ── cache helpers ─────────────────────────────────────────────────

def _cache_key(prompt: str, top_k: int) -> str:
    """SHA-256 of prompt+top_k → unique, collision-proof cache key."""
    raw = f"{prompt}|{top_k}"
    return "autocomplete:" + hashlib.sha256(raw.encode()).hexdigest()


async def _get_redis() -> Optional[aioredis.Redis]:
    global _redis
    if _redis is not None:
        return _redis
    try:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True,
                                   socket_connect_timeout=1)
        await _redis.ping()
        return _redis
    except Exception:
        _redis = None
        return None


async def cache_get(key: str) -> Optional[List[dict]]:
    r = await _get_redis()
    if r is None:
        return None
    try:
        val = await r.get(key)
        return json.loads(val) if val else None
    except Exception:
        return None


async def cache_set(key: str, value: List[dict]):
    r = await _get_redis()
    if r is None:
        return
    try:
        await r.setex(key, CACHE_TTL, json.dumps(value))
    except Exception:
        pass


# ── model inference ───────────────────────────────────────────────

def _run_inference(prompt: str, top_k: int) -> List[dict]:
    tok   = _tokenizer
    model = _model

    ids, type_ids = tok.encode_with_types(prompt)
    if not ids:
        return []

    src    = torch.tensor(ids,      dtype=torch.long).unsqueeze(1)
    type_t = torch.tensor(type_ids, dtype=torch.long).unsqueeze(1)

    with torch.no_grad():
        try:
            logits, _ = model(src, type_ids=type_t)
        except TypeError:
            logits, _ = model(src, None)

    probs = torch.softmax(logits[-1, 0], dim=-1)
    top_vals, top_idx = probs.topk(min(top_k * 2, tok.n_tokens))

    predictions = []
    seen = set()
    for prob, idx in zip(top_vals.tolist(), top_idx.tolist()):
        token_str = tok.itos[idx]
        if token_str.strip() and token_str not in seen:
            seen.add(token_str)
            predictions.append({"text": token_str, "prob": round(prob * 100, 1)})
        if len(predictions) >= top_k:
            break

    return predictions


# ═══════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def index():
    html = os.path.join(os.path.dirname(__file__), 'index.html')
    return FileResponse(html)


@app.get("/health", response_model=HealthResponse)
async def health():
    r = await _get_redis()
    cache_ok = False
    if r:
        try:
            await r.ping()
            cache_ok = True
        except Exception:
            pass
    return HealthResponse(
        ready=_ready,
        status=_status,
        model_info=_model_info,
        cache_online=cache_ok,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    total = _metrics["total_requests"]
    hits  = _metrics["cache_hits"]
    rate  = f"{hits/total*100:.1f}%" if total else "0.0%"
    avg_l = (_metrics["total_latency_ms"] / total) if total else 0.0
    return MetricsResponse(
        total_requests=total,
        cache_hits=hits,
        cache_misses=_metrics["cache_misses"],
        cache_hit_rate=rate,
        errors=_metrics["errors"],
        avg_latency_ms=f"{avg_l:.1f}ms",
    )


@app.post("/autocomplete", response_model=AutocompleteResponse)
async def autocomplete(req: AutocompleteRequest):
    if not _ready:
        return AutocompleteResponse(success=False, reason=_status)

    if not req.prompt.strip():
        return AutocompleteResponse(success=True, predictions=[])

    t0  = time.perf_counter()
    key = _cache_key(req.prompt, req.top_k)
    _metrics["total_requests"] += 1

    # ── cache check ───────────────────────────────────────────────
    cached_result = await cache_get(key)
    if cached_result is not None:
        _metrics["cache_hits"] += 1
        latency = (time.perf_counter() - t0) * 1000
        _metrics["total_latency_ms"] += latency
        return AutocompleteResponse(
            success=True,
            predictions=[Prediction(**p) for p in cached_result],
            cached=True,
            latency_ms=round(latency, 2),
        )

    # ── model inference ───────────────────────────────────────────
    _metrics["cache_misses"] += 1
    try:
        loop = asyncio.get_event_loop()
        with _model_lock:
            predictions = await loop.run_in_executor(
                None, _run_inference, req.prompt, req.top_k
            )

        # store in cache
        await cache_set(key, predictions)

        latency = (time.perf_counter() - t0) * 1000
        _metrics["total_latency_ms"] += latency

        return AutocompleteResponse(
            success=True,
            predictions=[Prediction(**p) for p in predictions],
            cached=False,
            latency_ms=round(latency, 2),
        )

    except Exception as e:
        _metrics["errors"] += 1
        return AutocompleteResponse(success=False, reason=str(e))


# ── run directly ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server with Redis caching…")
    print("API docs: http://localhost:5001/docs")
    print("Web UI:   http://localhost:5001")
    uvicorn.run("webapp.api:app", host="0.0.0.0", port=5001,
                reload=False, log_level="info")
