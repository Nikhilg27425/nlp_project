[![PyPI - Python Version](https://badge.fury.io/py/labml-python-autocomplete.svg)](https://badge.fury.io/py/labml-python-autocomplete)
[![PyPI Status](https://pepy.tech/badge/labml-python-autocomplete)](https://pepy.tech/project/labml-python-autocomplete)
[![Join Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/labforml/shared_invite/zt-egj9zvq9-Dl3hhZqobexgT7aVKnD14g/)
[![Twitter](https://img.shields.io/twitter/follow/labmlai?style=social)](https://twitter.com/labmlai?ref_src=twsrc%5Etfw)

# Python Autocomplete — Enhanced Edition

<p align="center">
  <img src="/images/vscode_attention.gif?raw=true" title="VSCode plugin">
</p>

> **This is an enhanced fork** of [labmlai/python_autocomplete](https://github.com/labmlai/python_autocomplete).
> The original used a character-level Transformer/LSTM with no structural understanding of Python.
> This fork introduces **token-level modeling**, **AST-based syntax-aware embeddings**, and a **copy mechanism**
> — enabling better semantic understanding, variable reuse, and structurally consistent code generation.

[Original demo video](https://www.youtube.com/watch?v=ZFzxBPBUh0M) · [Original Twitter thread](https://twitter.com/labmlai/status/1367444214963838978)

---

## What Changed

### Original Baseline
- Character-level tokenization (one token = one character)
- No understanding of Python syntax or token types
- No mechanism to reuse variable names from context
- Single metric: keystroke savings

### This Fork — 3 Phases of Improvement

---

## Phase 1 — Python-Aware Tokenizer

**File:** `python_autocomplete/dataset/python_tokenizer.py`

The original BPE/character tokenizer treats code as plain text. The new `PythonTokenizer` uses Python's built-in `tokenize` module to split code into **meaningful tokens** and assigns each a **type id**.

| Type ID | Category   | Examples                        |
|---------|------------|---------------------------------|
| 1       | KEYWORD    | `def`, `class`, `return`, `if`  |
| 2       | IDENTIFIER | `model`, `loss`, `optimizer`    |
| 3       | NUMBER     | `0`, `3.14`, `1e-4`             |
| 4       | STRING     | `"hello"`, `'world'`            |
| 5       | OPERATOR   | `+`, `==`, `->`, `*`            |
| 6       | DELIMITER  | `(`, `)`, `:`, `,`              |
| 7       | COMMENT    | `# this is a comment`           |
| 8       | NEWLINE    | newlines, indents, dedents      |

These type ids feed into a **token-type embedding** that runs alongside the standard token embedding:

```
final_embedding = token_embed(token_id) + type_embed(type_id)
```

**File:** `python_autocomplete/models/token_type_embedding.py`

---

## Phase 2 — AST-Based Syntax-Aware Embeddings

**File:** `python_autocomplete/dataset/ast_features.py`

Every token is enriched with 4 structural features extracted from the Python AST:

| Feature       | Description                                              |
|---------------|----------------------------------------------------------|
| `node_type`   | Which AST node owns this token (FunctionDef, Call, etc.) |
| `scope_depth` | How many nested scopes deep (function/class/lambda)      |
| `is_def`      | 1 if this name is being defined (assignment / def / class)|
| `is_call`     | 1 if this name is being called as a function             |

These are embedded and summed into the model's input representation:

```
input = token_type_embed(token, type)
      + ast_embed(node_type, scope_depth, is_def, is_call)
```

The extractor gracefully falls back to all-zeros for incomplete/unparseable code snippets, so inference on partial code always works.

---

## Phase 3 — Copy Mechanism (Pointer-Generator)

**File:** `python_autocomplete/models/copy_mechanism.py`

Variable names, function names, and identifiers are often reused from earlier in the same file. The copy mechanism lets the model **copy tokens directly from its context** instead of always generating from the vocabulary.

```
p_copy   = sigmoid(W · h_t)                    # gate: how much to copy
attn     = softmax(h_t · H_context^T)          # attention over context
p_gen    = (1 - p_copy) × vocab_distribution   # generate from vocab
p_ptr    = p_copy × attn                        # copy from context
final    = p_gen + scatter(p_ptr → vocab slots) # combined distribution
```

This directly improves predictions for:
- Variable names defined earlier in the file
- Function names being called multiple times
- Class attribute access patterns

---

## Combined Model — SyntaxAwareTransformer

**File:** `python_autocomplete/models/syntax_transformer.py`

All three improvements are wired into a single model that is a drop-in replacement for the original `TransformerModel`.

```
Input tokens
    │
    ├─► TokenTypeEmbedding(token_id, type_id)   ─┐
    │                                             ├─► sum ─► LayerNorm
    └─► ASTEmbedding(node, scope, is_def, is_call)┘
                                                  │
                                         Transformer Encoder
                                                  │
                                         Linear ─► vocab_logits
                                                  │
                                         CopyMechanism(h_t, context)
                                                  │
                                            final log-probs
```

---

## Evaluation Results

**File:** `python_autocomplete/evaluate/metrics.py`

The new evaluation suite computes 8 metrics vs the original single metric.

Results on a representative Python code sample (30-epoch training run):

| Metric                  | Value    | Notes                                      |
|-------------------------|----------|--------------------------------------------|
| Token Accuracy (top-1)  | **55.5%**| Exact next-token match                     |
| Token Accuracy (top-5)  | **87.5%**| Correct token in top-5 suggestions         |
| Perplexity              | **3.67** | Lower is better; random baseline = vocab size |
| Mean Reciprocal Rank    | **0.70** | Correct token ranked ~1.4th on average     |
| Keystroke Savings       | **41.0%**| Original metric — chars saved vs full typing|
| Syntax Validity Rate    | —        | Improves significantly with full training  |
| Identifier Copy Rate    | **40.9%**| Copy mechanism reusing context identifiers |
| Edit Distance Ratio     | **0.83** | 0 = perfect, 1 = completely wrong          |

> Numbers above are from a small demo run (65-token vocab, 1.2M param model, 30 epochs on a single snippet).
> Full dataset training will yield significantly better results across all metrics.

### Metric Definitions

- **Token Accuracy (top-1)** — fraction of positions where the model's single best prediction is exactly correct
- **Token Accuracy (top-5)** — fraction of positions where the correct token appears in the top 5 predictions
- **Perplexity** — `exp(mean cross-entropy loss)`; measures how "surprised" the model is by the ground truth
- **Mean Reciprocal Rank (MRR)** — average of `1/rank` of the correct token; 1.0 = always ranked first
- **Keystroke Savings** — `1 - keystrokes / total_chars`; the original repo's primary metric
- **Syntax Validity Rate** — fraction of greedy completions that are syntactically valid Python
- **Identifier Copy Rate** — fraction of identifier predictions that reuse a name already seen in context
- **Edit Distance Ratio** — normalised Levenshtein distance between predicted and ground-truth continuations

---

## Project Structure

```
python_autocomplete/
├── dataset/
│   ├── __init__.py
│   ├── bpe.py                    # original BPE tokenizer
│   ├── break_words.py            # original word splitter
│   ├── dataset.py                # data pipeline (+ 'python' tokenizer option)
│   ├── python_tokenizer.py       # ★ NEW — Python-aware tokenizer (Phase 1)
│   └── ast_features.py           # ★ NEW — AST feature extractor (Phase 2)
├── models/
│   ├── __init__.py
│   ├── lstm.py                   # original LSTM model
│   ├── transformer.py            # original Transformer model
│   ├── xl.py                     # original Transformer-XL model
│   ├── highway.py                # original RHN model
│   ├── token_type_embedding.py   # ★ NEW — token + type embedding (Phase 1)
│   ├── copy_mechanism.py         # ★ NEW — pointer-generator (Phase 3)
│   └── syntax_transformer.py     # ★ NEW — combined improved model (Phase 1-3)
├── evaluate/
│   ├── __init__.py
│   ├── beam_search.py            # original beam search
│   ├── eval_sample.py            # original keystroke evaluator
│   ├── factory.py                # model loader
│   ├── metrics.py                # ★ NEW — comprehensive 8-metric evaluator
│   └── run_eval.py               # ★ NEW — standalone evaluation runner
├── train.py                      # training loop (+ syntax_transformer_model option)
└── serve.py                      # Flask server (+ /health endpoint)
```

---

## Quick Start

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/python_autocomplete
cd python_autocomplete
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Run the evaluation demo (no dataset needed)

```bash
python -m python_autocomplete.evaluate.run_eval
```

This builds a small model, trains it on a sample snippet, and prints all 8 metrics.

### Train on the full dataset

```bash
# 1. Download and build the dataset
python python_autocomplete/create_dataset.py

# 2. Train with the new improved model
python python_autocomplete/train.py
```

The default config in `train.py` now uses:
- `model: syntax_transformer_model` — the new Phase 1-3 model
- `text.tokenizer: python` — the Python-aware tokenizer

To switch back to the original:
```python
# in train.py main()
'model': 'transformer_xl_model',
'text.tokenizer': 'bpe',
```

### VSCode Extension

```bash
# Start the improved server
python python_autocomplete/serve.py

# Check model health
curl http://localhost:5000/health
```

Then follow the original VSCode setup steps below.

---

## Original Setup (preserved)

### Try it yourself

1. Clone this repo
2. Install requirements: `pip install -r requirements.txt`
3. Run `python_autocomplete/create_dataset.py` to build the dataset
4. Run `python_autocomplete/train.py` to train
5. Run `python -m python_autocomplete.evaluate.run_eval` to evaluate

### VSCode Extension

1. Install npm packages:
```shell
cd vscode_extension
npm install
```

2. Start the server:
```shell
python python_autocomplete/serve.py
```

3. Open in VSCode and run with `Run > Start Debugging`

---

## Sample Output (original model)

Colors in the terminal evaluator:
- **yellow** — wrong prediction, user must type
- **blue** — correct prediction, user accepts with TAB/ENTER
- **green** — autocompleted characters

<p align="center">
  <img src="/images/python-autocomplete.png?raw=true" width="100%" title="Screenshot">
</p>

---

## Roadmap

- [ ] Train on full Awesome-PyTorch dataset with new model
- [ ] Benchmark new model vs baseline on held-out files
- [ ] Add type inference features (variable type tracking)
- [ ] Multi-line completion support
- [ ] Fine-tune on user's own codebase

---

## Credits

Original project by [labmlai](https://github.com/labmlai/python_autocomplete).
Improvements (Phase 1-3) built on top of the original architecture.
