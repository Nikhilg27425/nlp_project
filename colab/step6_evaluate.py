# ================================================================
# STEP 6 — Evaluate: Run all 8 metrics on the trained model
# ================================================================

import math, pickle, sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, '/content/nlp_project')

BASE_DIR  = Path('/content/drive/MyDrive/python_autocomplete')
CACHE_DIR = BASE_DIR / 'cache'
CKPT_DIR  = BASE_DIR / 'checkpoints'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── load tokenizer ────────────────────────────────────────────────
with open(CACHE_DIR / 'tokenizer.pkl', 'rb') as f:
    tok = pickle.load(f)
print(f"Tokenizer loaded  vocab={tok.n_tokens}")

# ── load best model ───────────────────────────────────────────────
ckpt_path = CKPT_DIR / 'best.pt'
if not ckpt_path.exists():
    ckpt_path = CKPT_DIR / 'latest.pt'

ckpt = torch.load(ckpt_path, map_location=device)
print(f"Checkpoint loaded : {ckpt_path.name}  (epoch {ckpt['epoch']})")

from labml_nn.transformers import TransformerConfigs
from python_autocomplete.models.syntax_transformer import SyntaxAwareTransformer

D_MODEL  = ckpt.get('d_model',  512)
N_LAYERS = ckpt.get('n_layers', 6)

t_conf = TransformerConfigs()
t_conf.d_model     = D_MODEL
t_conf.n_layers    = N_LAYERS
t_conf.n_src_vocab = tok.n_tokens
t_conf.n_tgt_vocab = tok.n_tokens
t_conf.dropout     = 0.0   # no dropout at eval

model = SyntaxAwareTransformer(
    n_tokens = tok.n_tokens,
    d_model  = D_MODEL,
    encoder  = t_conf.encoder,
    dropout  = 0.0,
    use_copy = True,
).to(device)

model.load_state_dict(ckpt['model'])
model.eval()
print(f"Model loaded  params={sum(p.numel() for p in model.parameters()):,}")

# ── evaluation sample ─────────────────────────────────────────────
# Use a real Python snippet not seen during training
EVAL_CODE = '''
import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.q_proj  = nn.Linear(d_model, d_model)
        self.k_proj  = nn.Linear(d_model, d_model)
        self.v_proj  = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out  = torch.matmul(attn, v)
        out  = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss   = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
'''

from python_autocomplete.evaluate.metrics import ModelEvaluator

print("\nRunning evaluation...")
evaluator = ModelEvaluator(model, tok, device=str(device), max_tokens=500)
results   = evaluator.evaluate_all(EVAL_CODE,
                                    run_syntax=True,
                                    run_copy=True,
                                    run_edit=True)
print("\n" + results.pretty())

# ── training curve ────────────────────────────────────────────────
if 'history' in ckpt and ckpt['history']['train_loss']:
    h = ckpt['history']
    print("\nTraining history:")
    print(f"  {'Epoch':<6} {'TrainLoss':>10} {'ValidLoss':>10} {'TrainAcc':>10} {'ValidAcc':>10}")
    print("  " + "-" * 50)
    for i, (tl, vl, ta, va) in enumerate(zip(
            h['train_loss'], h['valid_loss'],
            h['train_acc'],  h['valid_acc']), 1):
        print(f"  {i:<6} {tl:>10.4f} {vl:>10.4f} {ta:>9.2%} {va:>9.2%}")

print("\n✅ Step 6 complete — run step7_export.py next")
