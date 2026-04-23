# ================================================================
# STEP 1 — Setup: Install dependencies, mount Drive, clone repo
# ================================================================
# Paste this into the first Colab cell and run it.

import subprocess, sys

def pip(*args):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *args])

print("Installing dependencies...")
pip('torch', 'labml', 'labml-helpers', 'labml-nn', 'einops', 'flask', 'tqdm', 'requests')
print("✅ Dependencies installed")

# ── Mount Google Drive ────────────────────────────────────────────
from pathlib import Path

try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = Path('/content/drive/MyDrive/python_autocomplete')
    print("✅ Google Drive mounted")
except ImportError:
    BASE_DIR = Path('./training_output')
    print("ℹ️  Running locally — saving to ./training_output")

BASE_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR  = BASE_DIR / 'data'
CKPT_DIR  = BASE_DIR / 'checkpoints'
CACHE_DIR = BASE_DIR / 'cache'

for d in [DATA_DIR, CKPT_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Clone repo ────────────────────────────────────────────────────
REPO_URL = 'https://github.com/Nikhilg27425/nlp_project.git'
REPO_DIR = Path('/content/nlp_project')

if not REPO_DIR.exists():
    subprocess.run(
        ['git', 'clone', '-b', 'feature/improvements', REPO_URL, str(REPO_DIR)],
        check=True
    )
    print("✅ Repo cloned")
else:
    subprocess.run(['git', '-C', str(REPO_DIR), 'pull'], check=True)
    print("✅ Repo updated")

sys.path.insert(0, str(REPO_DIR))

# ── GPU check ─────────────────────────────────────────────────────
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice : {device}")
if device.type == 'cuda':
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU found — go to Runtime > Change runtime type > GPU")

print(f"\n📁 BASE_DIR  : {BASE_DIR}")
print(f"📁 DATA_DIR  : {DATA_DIR}")
print(f"📁 CKPT_DIR  : {CKPT_DIR}")
print(f"📁 CACHE_DIR : {CACHE_DIR}")
print("\n✅ Step 1 complete — run step2_dataset.py next")
