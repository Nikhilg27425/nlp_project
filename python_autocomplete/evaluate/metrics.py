"""
Comprehensive Evaluation Metrics
==================================
Computes all meaningful metrics for the autocomplete model — both the
original keystroke-savings metric and new ones added for our improvements.

Metrics
-------
1.  token_accuracy        — % of next-token predictions that are exactly correct
2.  top5_accuracy         — % where correct token is in top-5 predictions
3.  keystroke_savings     — % of characters saved vs typing everything manually
                            (original metric from the repo)
4.  perplexity            — exp(mean cross-entropy loss) — lower is better
5.  mean_reciprocal_rank  — MRR of the correct token in the ranked prediction list
6.  syntax_validity_rate  — % of model completions that are syntactically valid Python
7.  identifier_copy_rate  — % of identifier predictions that reuse a name from context
                            (measures copy mechanism effectiveness)
8.  edit_distance_ratio   — normalised edit distance between prediction and ground truth

Run standalone
--------------
    python -m python_autocomplete.evaluate.metrics
"""

import math
import io
import tokenize as _tokenize
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from python_autocomplete.dataset.python_tokenizer import (
    PythonTokenizer, tokenize_python, TOK_IDENT
)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    """Standard Levenshtein edit distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        ndp = [i + 1]
        for j, cb in enumerate(b):
            ndp.append(min(dp[j] + (0 if ca == cb else 1),
                           dp[j + 1] + 1,
                           ndp[-1] + 1))
        dp = ndp
    return dp[-1]


def _is_valid_python(code: str) -> bool:
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False


def _identifiers_in(text: str) -> set:
    """Return the set of identifier strings in *text*."""
    ids = set()
    try:
        for tok in _tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == _tokenize.NAME:
                ids.add(tok.string)
    except _tokenize.TokenError:
        pass
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResults:
    # core
    token_accuracy:       float = 0.0
    top5_accuracy:        float = 0.0
    perplexity:           float = 0.0
    mean_reciprocal_rank: float = 0.0
    # original
    keystroke_savings:    float = 0.0
    total_keystrokes:     int   = 0
    total_chars:          int   = 0
    # new
    syntax_validity_rate: float = 0.0
    identifier_copy_rate: float = 0.0
    edit_distance_ratio:  float = 0.0

    # raw accumulators (not printed)
    _n_tokens:      int   = field(default=0, repr=False)
    _n_top1:        int   = field(default=0, repr=False)
    _n_top5:        int   = field(default=0, repr=False)
    _sum_log_prob:  float = field(default=0.0, repr=False)
    _sum_rr:        float = field(default=0.0, repr=False)
    _n_syntax:      int   = field(default=0, repr=False)
    _n_valid_syn:   int   = field(default=0, repr=False)
    _n_copy_pred:   int   = field(default=0, repr=False)
    _n_copy_hit:    int   = field(default=0, repr=False)
    _sum_ed:        float = field(default=0.0, repr=False)
    _n_ed:          int   = field(default=0, repr=False)

    def finalise(self):
        if self._n_tokens:
            self.token_accuracy       = self._n_top1 / self._n_tokens
            self.top5_accuracy        = self._n_top5 / self._n_tokens
            self.perplexity           = math.exp(-self._sum_log_prob / self._n_tokens)
            self.mean_reciprocal_rank = self._sum_rr  / self._n_tokens
        if self.total_chars:
            self.keystroke_savings    = 1.0 - self.total_keystrokes / self.total_chars
        if self._n_syntax:
            self.syntax_validity_rate = self._n_valid_syn / self._n_syntax
        if self._n_copy_pred:
            self.identifier_copy_rate = self._n_copy_hit / self._n_copy_pred
        if self._n_ed:
            self.edit_distance_ratio  = self._sum_ed / self._n_ed

    def pretty(self) -> str:
        lines = [
            "=" * 55,
            "  EVALUATION RESULTS",
            "=" * 55,
            f"  {'Metric':<30} {'Value':>10}",
            "-" * 55,
            f"  {'Token Accuracy (top-1)':<30} {self.token_accuracy:>9.2%}",
            f"  {'Token Accuracy (top-5)':<30} {self.top5_accuracy:>9.2%}",
            f"  {'Perplexity':<30} {self.perplexity:>10.2f}",
            f"  {'Mean Reciprocal Rank':<30} {self.mean_reciprocal_rank:>10.4f}",
            "-" * 55,
            f"  {'Keystroke Savings':<30} {self.keystroke_savings:>9.2%}",
            f"  {'Total Keystrokes':<30} {self.total_keystrokes:>10,}",
            f"  {'Total Characters':<30} {self.total_chars:>10,}",
            "-" * 55,
            f"  {'Syntax Validity Rate':<30} {self.syntax_validity_rate:>9.2%}",
            f"  {'Identifier Copy Rate':<30} {self.identifier_copy_rate:>9.2%}",
            f"  {'Edit Distance Ratio':<30} {self.edit_distance_ratio:>10.4f}",
            "=" * 55,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# main evaluator
# ─────────────────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """
    Evaluates a trained model on a code string.

    Parameters
    ----------
    model       : the AutoregressiveModel (SyntaxAwareTransformer or original)
    tokenizer   : PythonTokenizer (or any Tokenizer with .encode / .itos)
    device      : torch device
    max_tokens  : cap evaluation at this many tokens (None = all)
    top_k       : beam width for keystroke-savings evaluation
    """

    def __init__(self, model, tokenizer, device='cpu',
                 max_tokens: Optional[int] = None,
                 top_k: int = 5):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device
        self.max_tokens = max_tokens
        self.top_k     = top_k
        self.softmax   = nn.Softmax(dim=-1)

    # ── token-level metrics (fast, no beam search) ────────────────────────────

    @torch.no_grad()
    def compute_token_metrics(self, code: str) -> EvalResults:
        """
        Compute token_accuracy, top5_accuracy, perplexity, MRR in one
        efficient forward pass over the whole sequence.
        """
        res = EvalResults()

        ids = self.tokenizer.encode(code)
        if len(ids) < 2:
            return res

        if self.max_tokens:
            ids = ids[:self.max_tokens + 1]

        # get type_ids if tokenizer supports it
        type_ids = None
        if hasattr(self.tokenizer, 'encode_with_types'):
            raw_ids, type_ids_list = self.tokenizer.encode_with_types(code)
            if self.max_tokens:
                type_ids_list = type_ids_list[:self.max_tokens + 1]
            type_ids = torch.tensor(type_ids_list, dtype=torch.long,
                                    device=self.device).unsqueeze(1)

        src = torch.tensor(ids, dtype=torch.long,
                           device=self.device).unsqueeze(1)   # [S, 1]

        # forward pass
        self.model.eval()
        if type_ids is not None:
            try:
                logits, _ = self.model(src[:-1], type_ids=type_ids[:-1])
            except TypeError:
                logits, _ = self.model(src[:-1], None)
        else:
            logits, _ = self.model(src[:-1], None)

        # logits: [S-1, 1, V]
        probs = self.softmax(logits[:, 0, :])   # [S-1, V]
        targets = src[1:, 0]                    # [S-1]

        n = probs.shape[0]
        res._n_tokens = n

        # top-1 accuracy
        top1 = probs.argmax(dim=-1)
        res._n_top1 = (top1 == targets).sum().item()

        # top-5 accuracy
        top5 = probs.topk(min(5, probs.shape[-1]), dim=-1).indices
        for i in range(n):
            if targets[i] in top5[i]:
                res._n_top5 += 1

        # perplexity (mean negative log-likelihood)
        target_probs = probs[torch.arange(n), targets]
        res._sum_log_prob = target_probs.log().sum().item()

        # MRR
        sorted_idx = probs.argsort(dim=-1, descending=True)
        for i in range(n):
            rank = (sorted_idx[i] == targets[i]).nonzero(as_tuple=True)[0]
            if len(rank):
                res._sum_rr += 1.0 / (rank[0].item() + 1)

        res.finalise()
        return res

    # ── keystroke savings (greedy, no beam search needed) ─────────────────────

    @torch.no_grad()
    def compute_keystroke_savings(self, code: str) -> Tuple[float, int, int]:
        """
        Simulates the autocomplete loop:
          - at each position, take the top-1 prediction
          - if it matches the next N chars, skip those N chars (1 keystroke)
          - otherwise the user types 1 char (1 keystroke)

        Returns (savings_ratio, keystrokes, total_chars)
        """
        ids = self.tokenizer.encode(code)
        if len(ids) < 2:
            return 0.0, len(ids), len(ids)

        if self.max_tokens:
            ids = ids[:self.max_tokens + 1]

        itos = self.tokenizer.itos
        total_chars  = sum(len(itos[t]) for t in ids[1:])
        keystrokes   = 0
        chars_saved  = 0

        self.model.eval()
        i = 0
        while i < len(ids) - 1:
            src = torch.tensor(ids[:i + 1], dtype=torch.long,
                               device=self.device).unsqueeze(1)
            logits, _ = self.model(src, None)
            pred_id = logits[-1, 0].argmax().item()
            pred_str = itos[pred_id]
            true_str = itos[ids[i + 1]]

            if pred_str == true_str:
                chars_saved += len(true_str)
                keystrokes  += 1
                i += 1
            else:
                keystrokes += 1
                i += 1

        savings = chars_saved / total_chars if total_chars else 0.0
        return savings, keystrokes, total_chars

    # ── syntax validity ───────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_syntax_validity(self, code: str,
                                 n_samples: int = 50,
                                 completion_len: int = 20) -> float:
        """
        At *n_samples* random positions in the code, generate *completion_len*
        tokens greedily and check if the resulting snippet is valid Python.
        Returns the fraction of valid completions.
        """
        import random
        ids = self.tokenizer.encode(code)
        if len(ids) < 10:
            return 0.0

        itos = self.tokenizer.itos
        valid = 0
        positions = random.sample(range(5, len(ids) - 1),
                                  min(n_samples, len(ids) - 6))

        self.model.eval()
        for pos in positions:
            generated = list(ids[:pos])
            for _ in range(completion_len):
                src = torch.tensor(generated, dtype=torch.long,
                                   device=self.device).unsqueeze(1)
                logits, _ = self.model(src, None)
                next_id = logits[-1, 0].argmax().item()
                generated.append(next_id)

            snippet = ''.join(itos[t] for t in generated)
            if _is_valid_python(snippet):
                valid += 1

        return valid / len(positions) if positions else 0.0

    # ── identifier copy rate ──────────────────────────────────────────────────

    @torch.no_grad()
    def compute_copy_rate(self, code: str) -> float:
        """
        For each identifier token in the sequence, check whether the model's
        top-1 prediction is an identifier that already appeared in the context.
        Returns the fraction of such "copy hits".
        """
        if not hasattr(self.tokenizer, 'encode_with_types'):
            return 0.0

        ids, type_ids_list = self.tokenizer.encode_with_types(code)
        if len(ids) < 2:
            return 0.0

        if self.max_tokens:
            ids = ids[:self.max_tokens + 1]
            type_ids_list = type_ids_list[:self.max_tokens + 1]

        itos = self.tokenizer.itos
        copy_preds = 0
        copy_hits  = 0

        self.model.eval()
        for i in range(1, len(ids)):
            # only evaluate at identifier positions
            if type_ids_list[i] != TOK_IDENT:
                continue

            src = torch.tensor(ids[:i], dtype=torch.long,
                               device=self.device).unsqueeze(1)
            type_t = torch.tensor(type_ids_list[:i], dtype=torch.long,
                                  device=self.device).unsqueeze(1)
            try:
                logits, _ = self.model(src, type_ids=type_t)
            except TypeError:
                logits, _ = self.model(src, None)

            pred_id  = logits[-1, 0].argmax().item()
            pred_str = itos[pred_id]

            # context identifiers = all identifiers seen so far
            context_ids = _identifiers_in(''.join(itos[t] for t in ids[:i]))

            copy_preds += 1
            if pred_str in context_ids:
                copy_hits += 1

        return copy_hits / copy_preds if copy_preds else 0.0

    # ── edit distance ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_edit_distance(self, code: str,
                               window: int = 10) -> float:
        """
        At each token position, predict the next *window* tokens greedily and
        compare to the ground truth using normalised edit distance.
        Samples every 10th position for speed.
        """
        ids = self.tokenizer.encode(code)
        if len(ids) < window + 2:
            return 0.0

        itos = self.tokenizer.itos
        total_ed = 0.0
        n_samples = 0

        self.model.eval()
        for i in range(0, len(ids) - window - 1, 10):
            # generate window tokens greedily
            generated = list(ids[:i + 1])
            for _ in range(window):
                src = torch.tensor(generated, dtype=torch.long,
                                   device=self.device).unsqueeze(1)
                logits, _ = self.model(src, None)
                generated.append(logits[-1, 0].argmax().item())

            pred_str = ''.join(itos[t] for t in generated[i + 1:])
            true_str = ''.join(itos[t] for t in ids[i + 1: i + 1 + window])

            max_len = max(len(pred_str), len(true_str), 1)
            total_ed += _levenshtein(pred_str, true_str) / max_len
            n_samples += 1

        return total_ed / n_samples if n_samples else 0.0

    # ── combined runner ───────────────────────────────────────────────────────

    def evaluate_all(self, code: str,
                     run_syntax: bool = True,
                     run_copy: bool = True,
                     run_edit: bool = True) -> EvalResults:
        """Run all metrics and return a combined EvalResults."""
        print("  [1/5] Token-level metrics (accuracy, perplexity, MRR)…")
        res = self.compute_token_metrics(code)

        print("  [2/5] Keystroke savings…")
        savings, ks, tc = self.compute_keystroke_savings(code)
        res.keystroke_savings = savings
        res.total_keystrokes  = ks
        res.total_chars       = tc

        if run_syntax:
            print("  [3/5] Syntax validity…")
            res.syntax_validity_rate = self.compute_syntax_validity(code)

        if run_copy:
            print("  [4/5] Identifier copy rate…")
            res.identifier_copy_rate = self.compute_copy_rate(code)

        if run_edit:
            print("  [5/5] Edit distance…")
            res.edit_distance_ratio = self.compute_edit_distance(code)

        return res
