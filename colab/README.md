# Colab Training Pipeline

Run these files **in order** in Google Colab:

| Step | File | What it does |
|------|------|-------------|
| 1 | `step1_setup.py` | Install deps, clone repo, mount Drive |
| 2 | `step2_dataset.py` | Download 80+ Python repos, build corpus |
| 3 | `step3_tokenizer.py` | Train Python-aware tokenizer (~8k vocab) |
| 4 | `step4_encode.py` | Encode corpus to token ids |
| 5 | `step5_train.py` | Train SyntaxAwareTransformer on GPU |
| 6 | `step6_evaluate.py` | Run all 8 metrics on trained model |
| 7 | `step7_export.py` | Export model for Web UI |

## How to use in Colab

In each Colab cell, run:
```python
exec(open('/content/nlp_project/colab/stepN_xxx.py').read())
```

Or copy-paste the file contents directly into cells.

## Expected results after full training (T4 GPU, 20 epochs)
- Token Accuracy (top-1): ~65-75%
- Token Accuracy (top-5): ~90-95%
- Perplexity: ~2.0-3.0
- Keystroke Savings: ~45-55%
- Training time: ~2-3 hours
