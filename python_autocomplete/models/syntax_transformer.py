"""
Syntax-Aware Transformer
=========================
Integrates three improvements over the baseline TransformerModel:

  1. Token-Type Embeddings   — keyword vs identifier vs operator, etc.
  2. AST Structural Embeddings — node type + scope depth + def/call flags
  3. Copy Mechanism           — pointer-generator for variable reuse

Architecture
------------

  Input tokens  ──► TokenTypeEmbedding(token_id, type_id)  ─┐
  AST features  ──► ASTEmbedding(node_type, scope, flags)  ─┤
                                                              ▼
                                                    sum → LayerNorm
                                                              │
                                                    Transformer Encoder
                                                              │
                                                    Linear → vocab_logits
                                                              │
                                                    CopyMechanism (optional)
                                                              │
                                                    final_logits

The model is fully backward-compatible: if type_ids / ast_features are not
supplied it behaves like the original TransformerModel.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from labml_helpers.module import Module
from labml_nn.transformers import Encoder
from labml_nn.transformers.utils import subsequent_mask

from python_autocomplete.models import AutoregressiveModel
from python_autocomplete.models.token_type_embedding import TokenTypeEmbedding
from python_autocomplete.models.copy_mechanism import CopyMechanism
from python_autocomplete.dataset.ast_features import N_AST_NODE_TYPES


# ── AST feature embedding ─────────────────────────────────────────────────────

class ASTEmbedding(nn.Module):
    """
    Embeds four AST features and sums them into a single d_model vector.

      node_type   : which AST node owns this token
      scope_depth : nesting depth (clamped to max_depth)
      is_def      : binary — token is a name being defined
      is_call     : binary — token is a function being called
    """

    MAX_SCOPE_DEPTH = 16

    def __init__(self, d_model: int, n_node_types: int = N_AST_NODE_TYPES):
        super().__init__()
        self.node_embed  = nn.Embedding(n_node_types,          d_model)
        self.scope_embed = nn.Embedding(self.MAX_SCOPE_DEPTH,  d_model)
        self.def_embed   = nn.Embedding(2, d_model)
        self.call_embed  = nn.Embedding(2, d_model)

        for emb in (self.node_embed, self.scope_embed,
                    self.def_embed, self.call_embed):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self,
                node_types:   torch.Tensor,
                scope_depths: torch.Tensor,
                is_def:       torch.Tensor,
                is_call:      torch.Tensor) -> torch.Tensor:
        """All inputs: [seq_len, batch] long tensors."""
        # clamp all ids to valid embedding range
        n_nodes = self.node_embed.num_embeddings
        node_types   = node_types.clamp(0, n_nodes - 1)
        scope_depths = scope_depths.clamp(0, self.MAX_SCOPE_DEPTH - 1)
        is_def       = is_def.clamp(0, 1)
        is_call      = is_call.clamp(0, 1)
        return (self.node_embed(node_types)
                + self.scope_embed(scope_depths)
                + self.def_embed(is_def)
                + self.call_embed(is_call))


# ── main model ────────────────────────────────────────────────────────────────

class SyntaxAwareTransformer(AutoregressiveModel):
    """
    Parameters
    ----------
    n_tokens        : vocabulary size
    d_model         : model dimension
    encoder         : labml_nn Transformer Encoder
    dropout         : dropout probability
    use_copy        : whether to enable the copy mechanism
    """

    def __init__(self,
                 n_tokens: int,
                 d_model: int,
                 encoder: Encoder,
                 dropout: float = 0.1,
                 use_copy: bool = True):
        super().__init__()

        self.d_model   = d_model
        self.n_tokens  = n_tokens
        self.use_copy  = use_copy

        # ── embeddings ───────────────────────────────────────────────────────
        self.token_type_embed = TokenTypeEmbedding(n_tokens, d_model,
                                                   dropout=dropout)
        self.ast_embed        = ASTEmbedding(d_model)
        self.embed_norm       = nn.LayerNorm(d_model)

        # ── transformer ──────────────────────────────────────────────────────
        self.encoder   = encoder
        self.src_mask: Optional[torch.Tensor] = None

        # ── output projection ────────────────────────────────────────────────
        self.fc = nn.Linear(d_model, n_tokens)

        # ── copy mechanism ───────────────────────────────────────────────────
        self.copy = CopyMechanism(d_model, n_tokens) if use_copy else None

    # ── forward ──────────────────────────────────────────────────────────────

    def __call__(self,
                 src: torch.Tensor,
                 state: Any = None,
                 type_ids: Optional[torch.Tensor] = None,
                 ast_node_types: Optional[torch.Tensor] = None,
                 ast_scope_depths: Optional[torch.Tensor] = None,
                 ast_is_def: Optional[torch.Tensor] = None,
                 ast_is_call: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, None]:
        """
        src              : [seq_len, batch]  token ids
        type_ids         : [seq_len, batch]  token type ids  (optional)
        ast_*            : [seq_len, batch]  AST feature ids (optional)

        Returns (logits [seq_len, batch, n_tokens], None)
        """
        seq_len = src.shape[0]

        # ── causal mask ──────────────────────────────────────────────────────
        if self.src_mask is None or self.src_mask.size(0) != seq_len:
            self.src_mask = subsequent_mask(seq_len).to(src.device)

        # ── build embedding ──────────────────────────────────────────────────
        x = self.token_type_embed(src, type_ids)          # [S, B, D]

        if ast_node_types is not None:
            ast_emb = self.ast_embed(ast_node_types,
                                     ast_scope_depths,
                                     ast_is_def,
                                     ast_is_call)         # [S, B, D]
            x = x + ast_emb

        x = self.embed_norm(x)

        # ── transformer encoder ──────────────────────────────────────────────
        enc_out = self.encoder(x, self.src_mask)          # [S, B, D]

        # ── vocab projection ─────────────────────────────────────────────────
        vocab_logits = self.fc(enc_out)                   # [S, B, V]

        # ── copy mechanism ───────────────────────────────────────────────────
        if self.copy is not None and self.training:
            # During training apply copy at every position
            # h_t: last position hidden state [B, D]
            h_t = enc_out[-1]                             # [B, D]
            last_logits = vocab_logits[-1]                # [B, V]
            # context = all encoder outputs transposed to [B, S, D]
            ctx_hidden = enc_out.permute(1, 0, 2)         # [B, S, D]
            ctx_ids    = src.permute(1, 0)                # [B, S]
            vocab_logits[-1] = self.copy(h_t, last_logits,
                                         ctx_hidden, ctx_ids)

        return vocab_logits, None

    # ── inference helper ─────────────────────────────────────────────────────

    def predict_next(self,
                     src: torch.Tensor,
                     type_ids: Optional[torch.Tensor] = None,
                     ast_features: Optional[dict] = None) -> torch.Tensor:
        """
        Convenience method for inference.
        Returns log-probabilities for the next token: [batch, n_tokens]
        """
        ast_kwargs = {}
        if ast_features:
            ast_kwargs = {
                'ast_node_types':   ast_features.get('node_types'),
                'ast_scope_depths': ast_features.get('scope_depths'),
                'ast_is_def':       ast_features.get('is_def'),
                'ast_is_call':      ast_features.get('is_call'),
            }
        logits, _ = self.__call__(src, type_ids=type_ids, **ast_kwargs)
        return torch.log_softmax(logits[-1], dim=-1)
