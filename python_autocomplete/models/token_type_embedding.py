"""
Token-Type-Aware Embedding
==========================
Combines a standard token embedding with a small token-type embedding so the
model can distinguish keywords from identifiers, operators, literals, etc.

  final_embedding = token_embed(token_id) + type_embed(type_id)

The type_ids tensor is optional; when absent the module behaves exactly like a
plain nn.Embedding (backward-compatible with the existing models).
"""

import torch
import torch.nn as nn
from typing import Optional

from python_autocomplete.dataset.python_tokenizer import N_TOKEN_TYPES


class TokenTypeEmbedding(nn.Module):
    """
    Parameters
    ----------
    n_tokens   : vocabulary size
    d_model    : embedding dimension
    n_types    : number of token-type categories (default = N_TOKEN_TYPES = 10)
    dropout    : dropout applied to the summed embedding
    """

    def __init__(self,
                 n_tokens: int,
                 d_model: int,
                 n_types: int = N_TOKEN_TYPES,
                 dropout: float = 0.1):
        super().__init__()
        self.token_embed = nn.Embedding(n_tokens, d_model)
        self.type_embed  = nn.Embedding(n_types,  d_model)
        self.dropout     = nn.Dropout(dropout)
        self.d_model     = d_model

        # initialise type embeddings small so they act as a residual signal
        nn.init.normal_(self.type_embed.weight, std=0.02)

    def forward(self,
                token_ids: torch.Tensor,
                type_ids:  Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        token_ids : [seq_len, batch]  (long)
        type_ids  : [seq_len, batch]  (long) — optional
        returns   : [seq_len, batch, d_model]
        """
        x = self.token_embed(token_ids)
        if type_ids is not None:
            x = x + self.type_embed(type_ids)
        return self.dropout(x)
