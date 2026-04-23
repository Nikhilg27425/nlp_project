# ================================================================
# STEP 7 — Export: Package model for the Web UI
# ================================================================
# After this step, download `export/model_bundle.pt` from Drive
# and place it in your local `webapp/` folder.
# The web server will auto-load it on startup.

import pickle, sys, shutil
from pathlib import Path

import torch

sys.path.insert(0, '/content/nlp_project')

BASE_DIR   = Path('/content/drive/MyDrive/python_autocomplete')
CACHE_DIR  = BASE_DIR / 'cache'
CKPT_DIR   = BASE_DIR / 'checkpoints'
EXPORT_DIR = BASE_DIR / 'export'
EXPORT_DIR.mkdir(exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── load tokenizer ────────────────────────────────────────────────
with open(CACHE_DIR / 'tokenizer.pkl', 'rb') as f:
    tok = pickle.load(f)

# ── load best checkpoint ──────────────────────────────────────────
ckpt_path = CKPT_DIR / 'best.pt'
if not ckpt_path.exists():
    ckpt_path = CKPT_DIR / 'latest.pt'

ckpt = torch.load(ckpt_path, map_location='cpu')
print(f"Loaded checkpoint: epoch {ckpt['epoch']}")

# ── rebuild model ─────────────────────────────────────────────────
from labml_nn.transformers import TransformerConfigs
from python_autocomplete.models.syntax_transformer import SyntaxAwareTransformer

D_MODEL  = ckpt.get('d_model',  512)
N_LAYERS = ckpt.get('n_layers', 6)

t_conf = TransformerConfigs()
t_conf.d_model     = D_MODEL
t_conf.n_layers    = N_LAYERS
t_conf.n_src_vocab = tok.n_tokens
t_conf.n_tgt_vocab = tok.n_tokens
t_conf.dropout     = 0.0

model = SyntaxAwareTransformer(
    n_tokens = tok.n_tokens,
    d_model  = D_MODEL,
    encoder  = t_conf.encoder,
    dropout  = 0.0,
    use_copy = True,
)
model.load_state_dict(ckpt['model'])
model.eval()

# ── save bundle ───────────────────────────────────────────────────
bundle = {
    'model_state':  model.state_dict(),
    'tokenizer':    tok,
    'd_model':      D_MODEL,
    'n_layers':     N_LAYERS,
    'vocab_size':   tok.n_tokens,
    'epoch':        ckpt['epoch'],
    'valid_loss':   ckpt.get('best_valid_loss'),
    'history':      ckpt.get('history', {}),
}

bundle_path = EXPORT_DIR / 'model_bundle.pt'
torch.save(bundle, bundle_path)
print(f"✅ Bundle saved: {bundle_path}")
print(f"   vocab_size : {tok.n_tokens}")
print(f"   d_model    : {D_MODEL}")
print(f"   n_layers   : {N_LAYERS}")
print(f"   epoch      : {ckpt['epoch']}")

# ── also save tokenizer separately (for quick loading) ────────────
tok_path = EXPORT_DIR / 'tokenizer.pkl'
with open(tok_path, 'wb') as f:
    pickle.dump(tok, f)
print(f"✅ Tokenizer saved: {tok_path}")

# ── download instructions ─────────────────────────────────────────
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NEXT STEPS — use the trained model in the Web UI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Download from Google Drive:
     MyDrive/python_autocomplete/export/model_bundle.pt

2. Place it in your local project:
     python_autocomplete/webapp/model_bundle.pt

3. The webapp/server.py will auto-detect and load it.
   It checks for model_bundle.pt before training from scratch.

4. Start the server:
     source venv/bin/activate
     python webapp/server.py

5. Open http://localhost:5001
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
print("✅ Step 7 complete — training pipeline done!")
