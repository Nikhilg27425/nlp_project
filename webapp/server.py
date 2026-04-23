"""
Web UI Server
=============
Self-contained Flask server that:
  - Serves the editor UI at  GET  /
  - Trains a demo model on startup (no checkpoint needed)
  - Provides autocomplete at  POST /autocomplete
  - Provides model info at    GET  /health

Run:
    python webapp/server.py
Then open http://localhost:5001
"""

import json
import os
import sys
import threading

import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify, send_from_directory

# make sure the package is importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from python_autocomplete.dataset.python_tokenizer import PythonTokenizer
from python_autocomplete.dataset.ast_features import ASTFeatureExtractor
from python_autocomplete.models.syntax_transformer import SyntaxAwareTransformer

# ── training corpus ───────────────────────────────────────────────────────────
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
        if epoch % 2 == 0:
            acc = trainer.evaluate(data, labels)
            print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.2%}")
    return model

def load_data(path):
    data = torch.load(path)
    return data

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
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
            batch_idx = indices[i:i + self.batch_size]
            yield self.data[batch_idx]

    def __len__(self):
        return len(self.data) // self.batch_size
'''

# ── global model state ────────────────────────────────────────────────────────
_model     = None
_tokenizer = None
_lock      = threading.Lock()
_ready     = False
_status    = "Initializing…"


def _build_and_train():
    global _model, _tokenizer, _ready, _status

    try:
        # ── try loading a pre-trained bundle first ────────────────
        bundle_path = os.path.join(os.path.dirname(__file__), 'model_bundle.pt')
        if os.path.exists(bundle_path):
            _status = "Loading pre-trained model…"
            import pickle
            bundle = torch.load(bundle_path, map_location='cpu')
            tok = bundle['tokenizer']

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
            _ready     = True
            _status    = (f"Ready (pre-trained) | vocab={tok.n_tokens} "
                          f"params={sum(p.numel() for p in model.parameters()):,} "
                          f"epoch={bundle.get('epoch','?')}")
            return

        # ── fallback: train a demo model on startup ───────────────
        _status = "Building tokenizer…"
        tok = PythonTokenizer()
        tok.train(TRAINING_CODE)

        _status = "Building model…"
        try:
            from labml_nn.transformers import TransformerConfigs
            t_conf = TransformerConfigs()
            t_conf.d_model     = 256
            t_conf.n_layers    = 4
            t_conf.n_src_vocab = tok.n_tokens
            t_conf.n_tgt_vocab = tok.n_tokens
            t_conf.dropout     = 0.1
            model = SyntaxAwareTransformer(
                n_tokens = tok.n_tokens,
                d_model  = 256,
                encoder  = t_conf.encoder,
                dropout  = 0.1,
                use_copy = True,
            )
        except Exception:
            # fallback: simple LSTM if transformer fails
            from python_autocomplete.models.lstm import LstmModel
            model = LstmModel(n_tokens=tok.n_tokens,
                              embedding_size=256,
                              hidden_size=512,
                              n_layers=2)

        _status = "Training model (this takes ~30s)…"
        ids, type_ids = tok.encode_with_types(TRAINING_CODE)
        extractor = ASTFeatureExtractor()
        feats = extractor.extract(TRAINING_CODE)
        feats.pad_or_trim(len(ids))

        mk = lambda lst: torch.tensor(lst, dtype=torch.long).unsqueeze(1)
        src      = mk(ids)
        type_t   = mk(type_ids)
        node_t   = mk(feats.node_types)
        scope_t  = mk(feats.scope_depths)
        def_t    = mk(feats.is_def)
        call_t   = mk(feats.is_call)

        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        loss_fn   = nn.CrossEntropyLoss()

        model.train()
        EPOCHS = 80
        for epoch in range(1, EPOCHS + 1):
            optimizer.zero_grad()
            try:
                logits, _ = model(src[:-1], type_ids=type_t[:-1],
                                  ast_node_types=node_t[:-1],
                                  ast_scope_depths=scope_t[:-1],
                                  ast_is_def=def_t[:-1],
                                  ast_is_call=call_t[:-1])
            except TypeError:
                logits, _ = model(src[:-1], None)

            target = src[1:, 0]
            loss = loss_fn(logits[:, 0, :], target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if epoch % 20 == 0:
                acc = (logits[:, 0, :].argmax(-1) == target).float().mean().item()
                _status = f"Training… epoch {epoch}/{EPOCHS}  loss={loss.item():.3f}  acc={acc:.1%}"

        model.eval()
        _tokenizer = tok
        _model     = model
        _ready     = True
        _status    = f"Ready  |  vocab={tok.n_tokens}  params={sum(p.numel() for p in model.parameters()):,}"

    except Exception as e:
        _status = f"Error during startup: {e}"
        raise


# start training in background thread
threading.Thread(target=_build_and_train, daemon=True).start()

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static')


@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')


@app.route('/health')
def health():
    return jsonify({'ready': _ready, 'status': _status})


@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    if not _ready:
        return jsonify({'success': False, 'reason': _status})

    data = request.get_json()
    prompt = data.get('prompt', '')
    if not prompt.strip():
        return jsonify({'success': True, 'predictions': []})

    with _lock:
        try:
            tok   = _tokenizer
            model = _model

            ids, type_ids = tok.encode_with_types(prompt)
            if not ids:
                return jsonify({'success': True, 'predictions': []})

            src    = torch.tensor(ids,      dtype=torch.long).unsqueeze(1)
            type_t = torch.tensor(type_ids, dtype=torch.long).unsqueeze(1)

            model.eval()
            with torch.no_grad():
                try:
                    logits, _ = model(src, type_ids=type_t)
                except TypeError:
                    logits, _ = model(src, None)

            # get top-8 next token probabilities
            probs    = torch.softmax(logits[-1, 0], dim=-1)
            top_vals, top_idx = probs.topk(min(8, tok.n_tokens))

            predictions = []
            seen = set()
            for prob, idx in zip(top_vals.tolist(), top_idx.tolist()):
                token_str = tok.itos[idx]
                if token_str.strip() and token_str not in seen:
                    seen.add(token_str)
                    predictions.append({
                        'text': token_str,
                        'prob': round(prob * 100, 1),
                    })

            return jsonify({'success': True, 'predictions': predictions})

        except Exception as e:
            return jsonify({'success': False, 'reason': str(e)})


if __name__ == '__main__':
    print("Starting Python Autocomplete Web UI…")
    print("Open http://localhost:5001 in your browser")
    app.run(host='0.0.0.0', port=5001, debug=False)
