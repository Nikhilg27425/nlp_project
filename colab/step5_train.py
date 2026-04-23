# ================================================================
# STEP 5 — Train: SyntaxAwareTransformer on GPU
# ================================================================

import math, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, '/content/nlp_project')

BASE_DIR  = Path('/content/drive/MyDrive/python_autocomplete')
CACHE_DIR = BASE_DIR / 'cache'
CKPT_DIR  = BASE_DIR / 'checkpoints'
CKPT_DIR.mkdir(exist_ok=True)

# ── load tokenizer & encoded data ────────────────────────────────
with open(CACHE_DIR / 'tokenizer.pkl', 'rb') as f:
    tok = pickle.load(f)

train_ids = np.load(str(CACHE_DIR / 'train_ids.npy'))
valid_ids  = np.load(str(CACHE_DIR / 'valid_ids.npy'))

print(f"Vocab      : {tok.n_tokens}")
print(f"Train toks : {len(train_ids):,}")
print(f"Valid toks : {len(valid_ids):,}")

# ── hyperparameters ───────────────────────────────────────────────
# T4 GPU (16GB) — safe defaults
D_MODEL    = 512
N_LAYERS   = 6
N_HEADS    = 8
D_FF       = 2048
DROPOUT    = 0.1
SEQ_LEN    = 256
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 3e-4
WARMUP     = 2000
GRAD_CLIP  = 1.0
LOG_EVERY  = 200   # batches
SAVE_EVERY = 2     # epochs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice : {device}")
if device.type == 'cuda':
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ── dataset ───────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, ids, seq_len):
        self.ids     = torch.from_numpy(ids).long()
        self.seq_len = seq_len
        self.n       = (len(ids) - 1) // seq_len

    def __len__(self):  return self.n

    def __getitem__(self, i):
        s = i * self.seq_len
        return self.ids[s:s+self.seq_len], self.ids[s+1:s+self.seq_len+1]


train_loader = DataLoader(SeqDataset(train_ids, SEQ_LEN), batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
valid_loader = DataLoader(SeqDataset(valid_ids,  SEQ_LEN), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

print(f"Train batches : {len(train_loader):,}")
print(f"Valid batches : {len(valid_loader):,}")

# ── build model ───────────────────────────────────────────────────
from labml_nn.transformers import TransformerConfigs
from python_autocomplete.models.syntax_transformer import SyntaxAwareTransformer

t_conf = TransformerConfigs()
t_conf.d_model     = D_MODEL
t_conf.n_layers    = N_LAYERS
t_conf.n_heads     = N_HEADS
t_conf.ffn.d_ff    = D_FF
t_conf.n_src_vocab = tok.n_tokens
t_conf.n_tgt_vocab = tok.n_tokens
t_conf.dropout     = DROPOUT

model = SyntaxAwareTransformer(
    n_tokens = tok.n_tokens,
    d_model  = D_MODEL,
    encoder  = t_conf.encoder,
    dropout  = DROPOUT,
    use_copy = True,
).to(device)

print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")

# ── optimizer & scheduler ─────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer,
                               T_max=EPOCHS * len(train_loader),
                               eta_min=1e-5)
loss_fn   = nn.CrossEntropyLoss()

# ── resume from checkpoint ────────────────────────────────────────
start_epoch     = 1
best_valid_loss = float('inf')
history         = {'train_loss':[], 'valid_loss':[], 'train_acc':[], 'valid_acc':[]}
latest_ckpt     = CKPT_DIR / 'latest.pt'

if latest_ckpt.exists():
    print("\nResuming from checkpoint...")
    ckpt = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch     = ckpt['epoch'] + 1
    best_valid_loss = ckpt.get('best_valid_loss', float('inf'))
    history         = ckpt.get('history', history)
    print(f"  Resumed from epoch {ckpt['epoch']}  best_loss={best_valid_loss:.4f}")

# ── training helpers ──────────────────────────────────────────────
global_step = (start_epoch - 1) * len(train_loader)

def warmup_lr(step):
    if step < WARMUP:
        for pg in optimizer.param_groups:
            pg['lr'] = LR * (step + 1) / WARMUP


def run_epoch(loader, train):
    global global_step
    model.train() if train else model.eval()
    tot_loss = tot_correct = tot_toks = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for step, (x, y) in enumerate(loader):
            # x,y: [B, S] → transpose to [S, B] for model
            x = x.t().contiguous().to(device)
            y = y.t().contiguous().to(device)

            logits, _ = model(x, None)          # [S, B, V]
            loss = loss_fn(logits.view(-1, tok.n_tokens), y.view(-1))

            if train:
                warmup_lr(global_step)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                global_step += 1

            n = y.numel()
            tot_loss    += loss.item() * n
            tot_correct += (logits.argmax(-1) == y).sum().item()
            tot_toks    += n

            if train and (step + 1) % LOG_EVERY == 0:
                avg_l = tot_loss / tot_toks
                avg_a = tot_correct / tot_toks
                lr_now = optimizer.param_groups[0]['lr']
                print(f"    step {step+1:5d}/{len(loader)}"
                      f"  loss={avg_l:.4f}"
                      f"  ppl={math.exp(min(avg_l,20)):.2f}"
                      f"  acc={avg_a:.2%}"
                      f"  lr={lr_now:.2e}")

    return tot_loss / tot_toks, tot_correct / tot_toks


# ── main loop ─────────────────────────────────────────────────────
print(f"\nTraining {EPOCHS} epochs on {device}")
print("=" * 60)

for epoch in range(start_epoch, EPOCHS + 1):
    t0 = time.time()
    print(f"\nEpoch {epoch}/{EPOCHS}")

    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(valid_loader, train=False)
    elapsed = time.time() - t0

    history['train_loss'].append(tr_loss)
    history['valid_loss'].append(vl_loss)
    history['train_acc'].append(tr_acc)
    history['valid_acc'].append(vl_acc)

    print(f"  train  loss={tr_loss:.4f}  ppl={math.exp(min(tr_loss,20)):.2f}  acc={tr_acc:.2%}")
    print(f"  valid  loss={vl_loss:.4f}  ppl={math.exp(min(vl_loss,20)):.2f}  acc={vl_acc:.2%}")
    print(f"  time   {elapsed:.0f}s")

    # save latest
    ckpt = dict(epoch=epoch, model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                best_valid_loss=best_valid_loss,
                history=history,
                vocab_size=tok.n_tokens,
                d_model=D_MODEL, n_layers=N_LAYERS)
    torch.save(ckpt, latest_ckpt)

    # save best
    if vl_loss < best_valid_loss:
        best_valid_loss = vl_loss
        torch.save(ckpt, CKPT_DIR / 'best.pt')
        print(f"  ⭐ New best model saved  (loss={best_valid_loss:.4f})")

    # periodic checkpoint
    if epoch % SAVE_EVERY == 0:
        torch.save(ckpt, CKPT_DIR / f'epoch_{epoch:02d}.pt')
        print(f"  💾 Checkpoint saved: epoch_{epoch:02d}.pt")

print("\n✅ Training complete!")
print(f"Best valid loss : {best_valid_loss:.4f}")
print(f"Best valid ppl  : {math.exp(min(best_valid_loss,20)):.2f}")
print("\n✅ Step 5 complete — run step6_evaluate.py next")
