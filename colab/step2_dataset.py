# ================================================================
# STEP 2 — Dataset: Download Python repos and build corpus
# ================================================================
# Requires step1_setup.py to have been run first.

import random
import re
import string
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

# paths set in step1
BASE_DIR  = Path('/content/drive/MyDrive/python_autocomplete')
DATA_DIR  = BASE_DIR / 'data'
CACHE_DIR = BASE_DIR / 'cache'

PRINTABLE = set(string.printable)

# ── repo lists ────────────────────────────────────────────────────
# High-quality Python repos — always included
CORE_REPOS = [
    ('numpy',        'numpy'),
    ('pandas-dev',   'pandas'),
    ('scikit-learn', 'scikit-learn'),
    ('keras-team',   'keras'),
    ('matplotlib',   'matplotlib'),
    ('requests',     'requests'),
    ('pallets',      'flask'),
    ('fastapi',      'fastapi'),
    ('pytest-dev',   'pytest'),
    ('psf',          'black'),
    ('PyCQA',        'flake8'),
    ('sqlalchemy',   'sqlalchemy'),
    ('celery',       'celery'),
    ('encode',       'httpx'),
    ('pydantic',     'pydantic'),
    ('tiangolo',     'sqlmodel'),
    ('aio-libs',     'aiohttp'),
    ('python-attrs', 'attrs'),
    ('more-itertools','more-itertools'),
    ('pytoolz',      'toolz'),
]


def get_awesome_pytorch_repos():
    try:
        url = ('https://raw.githubusercontent.com/'
               'bharathgs/Awesome-pytorch-list/master/README.md')
        with urllib.request.urlopen(url, timeout=15) as r:
            content = r.read().decode('utf-8', errors='ignore')
        pattern = re.compile(
            r'https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)')
        repos = list(set(pattern.findall(content)))
        print(f"  Found {len(repos)} repos in Awesome-PyTorch list")
        return repos
    except Exception as e:
        print(f"  Could not fetch list: {e}")
        return []


def download_repo(org, repo, dest_dir):
    extract_path = dest_dir / f'{org}_{repo}'
    if extract_path.exists():
        return extract_path

    for branch in ['main', 'master']:
        url = f'https://github.com/{org}/{repo}/archive/refs/heads/{branch}.zip'
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                data = r.read()
            zip_path = dest_dir / f'{org}_{repo}.zip'
            with open(zip_path, 'wb') as f:
                f.write(data)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_path)
            zip_path.unlink(missing_ok=True)
            return extract_path
        except Exception:
            continue
    return None


def read_py_file(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return ''.join(c for c in content if c in PRINTABLE)
    except Exception:
        return ''


# ── download ──────────────────────────────────────────────────────
print("Fetching Awesome-PyTorch repo list...")
extra = get_awesome_pytorch_repos()
all_repos = list(set(map(tuple, extra)) | set(CORE_REPOS))
random.shuffle(all_repos)
all_repos = all_repos[:80]   # cap at 80 to keep download time ~15 min

download_dir = DATA_DIR / 'downloads'
download_dir.mkdir(exist_ok=True)

all_py_files = []
failed = 0
for org, repo in tqdm(all_repos, desc='Downloading repos'):
    path = download_repo(org, repo, download_dir)
    if path:
        files = [p for p in path.rglob('*.py')
                 if p.is_file() and p.stat().st_size < 300_000]
        all_py_files.extend(files)
    else:
        failed += 1

print(f"\n✅ {len(all_py_files):,} Python files collected  ({failed} repos failed)")

# ── build train / valid split ─────────────────────────────────────
random.shuffle(all_py_files)
split       = int(len(all_py_files) * 0.9)
train_files = all_py_files[:split]
valid_files = all_py_files[split:]

TRAIN_PATH = DATA_DIR / 'train.py'
VALID_PATH = DATA_DIR / 'valid.py'

MAX_TRAIN = 50_000_000   # 50 MB
MAX_VALID =  5_000_000   #  5 MB


def write_corpus(files, out_path, max_chars):
    total = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for p in tqdm(files, desc=f'Writing {out_path.name}'):
            if total >= max_chars:
                break
            content = read_py_file(p)
            if len(content) < 50:
                continue
            f.write(f'\n# FILE: {p.name}\n')
            f.write(content + '\n')
            total += len(content)
    return total


tc = write_corpus(train_files, TRAIN_PATH, MAX_TRAIN)
vc = write_corpus(valid_files, VALID_PATH, MAX_VALID)

print(f"\n✅ Train corpus : {tc/1e6:.1f} MB")
print(f"✅ Valid corpus : {vc/1e6:.1f} MB")
print("\n✅ Step 2 complete — run step3_tokenizer.py next")
