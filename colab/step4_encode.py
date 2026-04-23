# ================================================================
# STEP 4 — Encode: Convert corpus text → token id arrays
# ================================================================

import pickle, sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/content/nlp_project')

BASE_DIR  = Path('/content/drive/MyDrive/python_autocomplete')
DATA_DIR  = BASE_DIR / 'data'
CACHE_DIR = BASE_DIR / 'cache'

# load tokenizer from step 3
with open(CACHE_DIR / 'tokenizer.pkl', 'rb') as f:
    tok = pickle.load(f)
print(f"Tokenizer loaded  vocab={tok.n_tokens}")

TRAIN_IDS = CACHE_DIR / 'train_ids.npy'
VALID_IDS = CACHE_DIR / 'valid_ids.npy'

# ── encode in chunks to avoid OOM ────────────────────────────────
CHUNK = 500_000   # characters per chunk

def encode_file(txt_path, cache_path, max_chars=None):
    if cache_path.exists():
        print(f"  Loading cached {cache_path.name}...")
        return np.load(str(cache_path))

    print(f"  Encoding {txt_path.name}...")
    all_ids = []

    with open(txt_path, 'r', encoding='utf-8') as f:
        chars_read = 0
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            ids = tok.encode(chunk, is_silent=True)
            all_ids.extend(ids)
            chars_read += len(chunk)
            print(f"    {chars_read/1e6:.1f}MB  →  {len(all_ids):,} tokens", end='\r')
            if max_chars and chars_read >= max_chars:
                break

    arr = np.array(all_ids, dtype=np.int32)
    np.save(str(cache_path), arr)
    print(f"\n  Saved {len(arr):,} tokens → {cache_path.name}")
    return arr


train_ids = encode_file(DATA_DIR / 'train.py', TRAIN_IDS, max_chars=20_000_000)
valid_ids  = encode_file(DATA_DIR / 'valid.py', VALID_IDS, max_chars=2_000_000)

print(f"\n✅ Train tokens : {len(train_ids):,}  ({len(train_ids)/1e6:.1f}M)")
print(f"✅ Valid tokens : {len(valid_ids):,}  ({len(valid_ids)/1e6:.1f}M)")
print(f"✅ Vocab size   : {tok.n_tokens}")
print("\n✅ Step 4 complete — run step5_train.py next")
