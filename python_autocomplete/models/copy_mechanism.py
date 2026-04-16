"""
Copy Mechanism (Pointer-Generator)
===================================
Allows the model to either *generate* the next token from the vocabulary
distribution OR *copy* a token that appeared earlier in the input context.

Architecture
------------
  p_copy  = sigmoid(W_copy · h_t)          # scalar gate: copy vs generate
  attn    = softmax(h_t · H_context^T)     # attention over context tokens
  p_gen   = (1 - p_copy) * vocab_dist      # generation distribution
  p_ptr   = p_copy * attn                  # pointer distribution over context

  final_logits[v] = p_gen[v] + sum_{t: context[t]==v} p_ptr[t]

The module is designed to wrap any existing AutoregressiveModel output.

Usage
-----
    copy = CopyMechanism(d_model, n_tokens)
    final_logits = copy(h_t, vocab_logits, context_hidden, context_token_ids)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CopyMechanism(nn.Module):
    """
    Parameters
    ----------
    d_model  : hidden dimension of the base model
    n_tokens : vocabulary size
    """

    def __init__(self, d_model: int, n_tokens: int):
        super().__init__()
        self.n_tokens = n_tokens

        # gate: decides how much to copy vs generate
        self.copy_gate = nn.Linear(d_model, 1)

        # attention projection for pointer network
        self.attn_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self,
                h_t: torch.Tensor,
                vocab_logits: torch.Tensor,
                context_hidden: Optional[torch.Tensor],
                context_token_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        h_t               : [batch, d_model]   current hidden state
        vocab_logits      : [batch, n_tokens]  raw logits from base model
        context_hidden    : [batch, ctx_len, d_model]  encoder hidden states
                            (None → pure generation, no copy)
        context_token_ids : [batch, ctx_len]   token ids of context tokens
                            (None → pure generation)

        Returns
        -------
        final_logits : [batch, n_tokens]  (NOT softmaxed — use with cross-entropy)
        """
        if context_hidden is None or context_token_ids is None:
            return vocab_logits

        batch, ctx_len, d_model = context_hidden.shape

        # ── generation distribution ──────────────────────────────────────────
        p_gen_dist = F.softmax(vocab_logits, dim=-1)          # [B, V]

        # ── copy gate ────────────────────────────────────────────────────────
        p_copy = torch.sigmoid(self.copy_gate(h_t))           # [B, 1]

        # ── pointer attention ────────────────────────────────────────────────
        # project h_t for attention query
        query = self.attn_proj(h_t).unsqueeze(2)              # [B, d, 1]
        # context_hidden: [B, ctx_len, d]
        attn_scores = torch.bmm(context_hidden, query).squeeze(2)  # [B, ctx_len]
        attn_weights = F.softmax(attn_scores, dim=-1)              # [B, ctx_len]

        # ── scatter pointer weights onto vocabulary ───────────────────────────
        # For each context position, add its attention weight to the
        # corresponding vocabulary slot.
        ptr_dist = torch.zeros_like(p_gen_dist)               # [B, V]
        # context_token_ids: [B, ctx_len]
        ptr_dist.scatter_add_(1, context_token_ids, attn_weights)

        # ── combine ──────────────────────────────────────────────────────────
        combined = (1.0 - p_copy) * p_gen_dist + p_copy * ptr_dist  # [B, V]

        # Return log-space for numerical stability with NLLLoss,
        # but keep as logits-compatible by returning log(combined + eps)
        return torch.log(combined + 1e-10)

    # ── convenience: loss function compatible with this output ───────────────

    @staticmethod
    def loss(log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """NLL loss — use when copy mechanism is active."""
        return F.nll_loss(log_probs.view(-1, log_probs.size(-1)), targets.view(-1))
