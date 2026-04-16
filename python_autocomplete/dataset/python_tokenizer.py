"""
Python-Aware Tokenizer
======================
Replaces the generic BPE/character tokenizer with a tokenizer that understands
Python's actual token types (keyword, identifier, operator, literal, etc.).

Each token carries:
  - token string  (what the model predicts)
  - token type id (fed as a second embedding for syntax awareness)

Token type categories (mapped to small int ids):
  0  UNKNOWN / OTHER
  1  KEYWORD        (if, def, class, return, …)
  2  IDENTIFIER     (variable / function names)
  3  NUMBER         (int / float literals)
  4  STRING         (string literals)
  5  OPERATOR       (+ - * / = == != …)
  6  DELIMITER      (: , ; ( ) [ ] { })
  7  COMMENT        (# …)
  8  NEWLINE / INDENT / DEDENT
  9  WHITESPACE
"""

import io
import keyword
import tokenize as _tokenize
from typing import Dict, List, Tuple

from python_autocomplete.dataset import ID_CHARS, Tokenizer

# ── token-type constants ──────────────────────────────────────────────────────
TOK_OTHER     = 0
TOK_KEYWORD   = 1
TOK_IDENT     = 2
TOK_NUMBER    = 3
TOK_STRING    = 4
TOK_OPERATOR  = 5
TOK_DELIMITER = 6
TOK_COMMENT   = 7
TOK_NEWLINE   = 8
TOK_WHITESPACE = 9

N_TOKEN_TYPES = 10

_DELIMITERS = set('()[]{}:,;@')
_OPERATORS  = set('+-*/%&|^~<>=!.')

_PY_KEYWORDS = set(keyword.kwlist)


def _classify(tok_type: int, tok_string: str) -> int:
    """Map a stdlib tokenize type + string to our compact type id."""
    if tok_type == _tokenize.NAME:
        return TOK_KEYWORD if tok_string in _PY_KEYWORDS else TOK_IDENT
    if tok_type == _tokenize.NUMBER:
        return TOK_NUMBER
    if tok_type == _tokenize.STRING:
        return TOK_STRING
    if tok_type == _tokenize.COMMENT:
        return TOK_COMMENT
    if tok_type in (_tokenize.NEWLINE, _tokenize.NL,
                    _tokenize.INDENT, _tokenize.DEDENT):
        return TOK_NEWLINE
    if tok_type == _tokenize.OP:
        if tok_string in _DELIMITERS:
            return TOK_DELIMITER
        return TOK_OPERATOR
    if tok_type == _tokenize.ERRORTOKEN and tok_string.strip() == '':
        return TOK_WHITESPACE
    return TOK_OTHER


# ── low-level tokenisation helper ────────────────────────────────────────────

def tokenize_python(code: str) -> List[Tuple[str, int]]:
    """
    Tokenise *code* and return a list of (token_string, type_id) pairs.
    Falls back to character-level splitting on tokenisation errors so the
    function never raises.
    """
    results: List[Tuple[str, int]] = []
    try:
        tokens = _tokenize.generate_tokens(io.StringIO(code).readline)
        for tok in tokens:
            if tok.type in (_tokenize.ENCODING, _tokenize.ENDMARKER):
                continue
            s = tok.string
            if not s:
                continue
            results.append((s, _classify(tok.type, s)))
    except _tokenize.TokenError:
        # Partial / broken code — fall back to char-level
        for ch in code:
            results.append((ch, TOK_OTHER))
    return results


# ── PythonTokenizer ───────────────────────────────────────────────────────────

class PythonTokenizer(Tokenizer):
    """
    Token-level tokenizer that is aware of Python syntax.

    Vocabulary is built from the actual tokens seen in the training corpus
    (not BPE merges).  Each entry in *itos* is a token string; the parallel
    *type_itos* list holds the corresponding type id.

    The tokenizer exposes two encode paths:
      encode(text)            → List[int]   (token ids, compatible with base class)
      encode_with_types(text) → (List[int], List[int])  (token ids + type ids)
    """

    def __init__(self):
        self._itos: List[str] = []
        self._stoi: Dict[str, int] = {}
        self._type_itos: List[int] = []   # type id for each vocab entry
        self.is_trained: bool = False

    # ── Tokenizer interface ───────────────────────────────────────────────────

    @property
    def n_tokens(self) -> int:
        return len(self._itos)

    @property
    def itos(self) -> List[str]:
        return self._itos

    @property
    def stoi(self) -> Dict[str, int]:
        return self._stoi

    @property
    def type_itos(self) -> List[int]:
        """type_id for each vocab index — use for embedding lookup."""
        return self._type_itos

    def train(self, data: str):
        """Build vocabulary from *data* (the full training corpus string)."""
        print("[PythonTokenizer] Building vocabulary …")
        pairs = tokenize_python(data)

        # collect unique tokens preserving first-seen type
        seen: Dict[str, int] = {}   # token_str → type_id
        for tok_str, tok_type in pairs:
            if tok_str not in seen:
                seen[tok_str] = tok_type

        # sort for determinism: keywords first, then identifiers, then rest
        ordered = sorted(seen.items(), key=lambda x: (x[1], x[0]))

        self._itos = [s for s, _ in ordered]
        self._stoi = {s: i for i, s in enumerate(self._itos)}
        self._type_itos = [t for _, t in ordered]
        self.is_trained = True
        print(f"[PythonTokenizer] Vocabulary size: {self.n_tokens}")

    def encode(self, data: str, *, is_silent: bool = True) -> List[int]:
        pairs = tokenize_python(data)
        ids = []
        for tok_str, _ in pairs:
            if tok_str in self._stoi:
                ids.append(self._stoi[tok_str])
            else:
                # unknown token → char-level fallback
                for ch in tok_str:
                    if ch in self._stoi:
                        ids.append(self._stoi[ch])
        return ids

    def encode_with_types(self, data: str) -> Tuple[List[int], List[int]]:
        """Return (token_ids, type_ids) in parallel."""
        pairs = tokenize_python(data)
        tok_ids, type_ids = [], []
        for tok_str, tok_type in pairs:
            if tok_str in self._stoi:
                tok_ids.append(self._stoi[tok_str])
                type_ids.append(tok_type)
            else:
                for ch in tok_str:
                    if ch in self._stoi:
                        tok_ids.append(self._stoi[ch])
                        type_ids.append(tok_type)
        return tok_ids, type_ids

    def rstrip(self, data: str) -> Tuple[str, List[int]]:
        """
        Strip the last (incomplete) token and return
        (stripped_string, encoded_ids_of_stripped_string).
        """
        pairs = tokenize_python(data)
        if not pairs:
            return '', []
        # drop the last token (it may be incomplete)
        complete_pairs = pairs[:-1]
        stripped = ''.join(s for s, _ in complete_pairs)
        ids = []
        for tok_str, _ in complete_pairs:
            if tok_str in self._stoi:
                ids.append(self._stoi[tok_str])
            else:
                for ch in tok_str:
                    if ch in self._stoi:
                        ids.append(self._stoi[ch])
        return stripped, ids

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self) -> dict:
        return {
            'itos': self._itos,
            'type_itos': self._type_itos,
        }

    def load(self, itos: List[str], type_itos: List[int]):
        self._itos = itos
        self._stoi = {s: i for i, s in enumerate(itos)}
        self._type_itos = type_itos
        self.is_trained = True
