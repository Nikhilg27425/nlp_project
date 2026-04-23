# ================================================================
# STEP 3 — Tokenizer: Build Python-aware vocabulary
# ================================================================

import pickle, sys
from pathlib import Path

sys.path.insert(0, '/content/nlp_project')

BASE_DIR  = Path('/content/drive/MyDrive/python_autocomplete')
DATA_DIR  = BASE_DIR / 'data'
CACHE_DIR = BASE_DIR / 'cache'

from python_autocomplete.dataset.python_tokenizer import PythonTokenizer

TOK_CACHE = CACHE_DIR / 'tokenizer.pkl'

if TOK_CACHE.exists():
    print("Loading cached tokenizer...")
    with open(TOK_CACHE, 'rb') as f:
        tok = pickle.load(f)
    print(f"✅ Loaded  vocab={tok.n_tokens}")
else:
    print("Training tokenizer on corpus...")
    # 5MB sample is enough to capture all real Python tokens
    with open(DATA_DIR / 'train.py', 'r', encoding='utf-8') as f:
        sample = f.read(5_000_000)

    tok = PythonTokenizer()
    tok.train(sample)

    with open(TOK_CACHE, 'wb') as f:
        pickle.dump(tok, f)
    print(f"✅ Tokenizer saved  vocab={tok.n_tokens}")

# ── vocab stats ───────────────────────────────────────────────────
from python_autocomplete.dataset.python_tokenizer import (
    TOK_KEYWORD, TOK_IDENT, TOK_NUMBER, TOK_STRING,
    TOK_OPERATOR, TOK_DELIMITER, TOK_COMMENT, TOK_NEWLINE
)

type_names = {
    TOK_KEYWORD:   'KEYWORD',
    TOK_IDENT:     'IDENTIFIER',
    TOK_NUMBER:    'NUMBER',
    TOK_STRING:    'STRING',
    TOK_OPERATOR:  'OPERATOR',
    TOK_DELIMITER: 'DELIMITER',
    TOK_COMMENT:   'COMMENT',
    TOK_NEWLINE:   'NEWLINE',
}

from collections import Counter
type_counts = Counter(tok.type_itos)

print(f"\nVocabulary breakdown ({tok.n_tokens} total tokens):")
print(f"  {'Type':<12} {'Count':>6}")
print("  " + "-" * 20)
for tid, name in type_names.items():
    print(f"  {name:<12} {type_counts[tid]:>6}")

print(f"\nSample keywords   : {[t for t,tp in zip(tok.itos, tok.type_itos) if tp==TOK_KEYWORD][:15]}")
print(f"Sample identifiers: {[t for t,tp in zip(tok.itos, tok.type_itos) if tp==TOK_IDENT][:15]}")
print(f"Sample operators  : {[t for t,tp in zip(tok.itos, tok.type_itos) if tp==TOK_OPERATOR][:15]}")

print("\n✅ Step 3 complete — run step4_encode.py next")
