"""
Standalone Evaluation Runner
==============================
Builds a small model, trains it on a sample snippet, then runs all metrics.

Usage (from repo root, with venv active):
    python -m python_autocomplete.evaluate.run_eval
"""

import torch
import torch.nn as nn
import torch.optim as optim

from labml_nn.transformers import TransformerConfigs

from python_autocomplete.dataset.python_tokenizer import PythonTokenizer
from python_autocomplete.dataset.ast_features import ASTFeatureExtractor, N_AST_NODE_TYPES
from python_autocomplete.models.syntax_transformer import SyntaxAwareTransformer
from python_autocomplete.evaluate.metrics import ModelEvaluator

# ── sample code used for both training and evaluation ────────────────────────
SAMPLE_CODE = '''
import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x

def train(model, data, labels, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn   = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss   = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
    return model

def evaluate(model, data, labels):
    with torch.no_grad():
        output = model(data)
        pred   = output.argmax(dim=-1)
        acc    = (pred == labels).float().mean()
    return acc.item()
'''


def build_and_train(code: str, d_model=128, n_layers=2,
                    epochs=30, lr=1e-3, device='cpu'):
    """Quick training loop so the model learns something meaningful."""

    # ── tokenizer ────────────────────────────────────────────────────────────
    tok = PythonTokenizer()
    tok.train(code)
    print(f"Vocabulary size : {tok.n_tokens}")

    # ── model ─────────────────────────────────────────────────────────────────
    t_conf = TransformerConfigs()
    t_conf.d_model     = d_model
    t_conf.n_layers    = n_layers
    t_conf.n_src_vocab = tok.n_tokens
    t_conf.n_tgt_vocab = tok.n_tokens
    t_conf.dropout     = 0.1

    model = SyntaxAwareTransformer(
        n_tokens = tok.n_tokens,
        d_model  = d_model,
        encoder  = t_conf.encoder,
        dropout  = 0.1,
        use_copy = True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params    : {total_params:,}")

    # ── encode with AST features ──────────────────────────────────────────────
    ids, type_ids = tok.encode_with_types(code)
    extractor = ASTFeatureExtractor()
    feats = extractor.extract(code)
    feats.pad_or_trim(len(ids))

    SEQ = len(ids)
    src      = torch.tensor(ids,                dtype=torch.long, device=device).unsqueeze(1)
    type_t   = torch.tensor(type_ids,           dtype=torch.long, device=device).unsqueeze(1)
    node_t   = torch.tensor(feats.node_types,   dtype=torch.long, device=device).unsqueeze(1)
    scope_t  = torch.tensor(feats.scope_depths, dtype=torch.long, device=device).unsqueeze(1)
    isdef_t  = torch.tensor(feats.is_def,       dtype=torch.long, device=device).unsqueeze(1)
    iscall_t = torch.tensor(feats.is_call,      dtype=torch.long, device=device).unsqueeze(1)

    target = src[1:, 0]   # [S-1]

    # ── training loop ─────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()

    print(f"\nTraining for {epochs} epochs…")
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits, _ = model(src[:-1], type_ids=type_t[:-1],
                          ast_node_types=node_t[:-1],
                          ast_scope_depths=scope_t[:-1],
                          ast_is_def=isdef_t[:-1],
                          ast_is_call=iscall_t[:-1])
        loss = loss_fn(logits[:, 0, :], target)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            acc = (logits[:, 0, :].argmax(-1) == target).float().mean().item()
            print(f"  epoch {epoch:3d}  loss={loss.item():.4f}  train_acc={acc:.2%}")

    return model, tok


def main():
    device = 'cpu'

    print("=" * 55)
    print("  BUILDING & TRAINING MODEL")
    print("=" * 55)
    model, tok = build_and_train(SAMPLE_CODE, device=device)

    print("\n" + "=" * 55)
    print("  RUNNING EVALUATION")
    print("=" * 55)
    evaluator = ModelEvaluator(model, tok, device=device, max_tokens=200)
    results = evaluator.evaluate_all(
        SAMPLE_CODE,
        run_syntax=True,
        run_copy=True,
        run_edit=True,
    )

    print("\n" + results.pretty())


if __name__ == '__main__':
    main()
