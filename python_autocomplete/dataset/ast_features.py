"""
AST Feature Extraction
======================
Parses Python source code with the built-in `ast` module and produces
per-token structural features that can be fed as additional embeddings.

Features extracted per token position
--------------------------------------
  ast_node_type  : integer id of the AST node that "owns" this token
                   (e.g. FunctionDef=1, ClassDef=2, Assign=3, …)
  scope_depth    : how many nested scopes (function/class/lambda) deep
  is_definition  : 1 if the token is a name being *defined* (def/class/assign lhs)
  is_call        : 1 if the token is a function being *called*

All features are returned as parallel integer lists aligned with the token
sequence produced by python_tokenizer.tokenize_python().

Usage
-----
    from python_autocomplete.dataset.ast_features import ASTFeatureExtractor
    extractor = ASTFeatureExtractor()
    features  = extractor.extract(source_code)
    # features.node_types  : List[int]
    # features.scope_depths: List[int]
    # features.is_def      : List[int]
    # features.is_call     : List[int]
"""

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from python_autocomplete.dataset.python_tokenizer import tokenize_python

# ── AST node-type registry ────────────────────────────────────────────────────

_NODE_REGISTRY: Dict[str, int] = {}
_NODE_COUNTER = [1]   # 0 = UNKNOWN

def _node_id(name: str) -> int:
    if name not in _NODE_REGISTRY:
        _NODE_REGISTRY[name] = _NODE_COUNTER[0]
        _NODE_COUNTER[0] += 1
    return _NODE_REGISTRY[name]

# Pre-register the most common nodes so their ids are stable across runs
for _n in ['Module','FunctionDef','AsyncFunctionDef','ClassDef','Return',
           'Assign','AugAssign','AnnAssign','For','AsyncFor','While','If',
           'With','AsyncWith','Raise','Try','Import','ImportFrom','Global',
           'Nonlocal','Expr','Pass','Break','Continue','BoolOp','BinOp',
           'UnaryOp','Lambda','IfExp','Dict','Set','ListComp','SetComp',
           'DictComp','GeneratorExp','Await','Yield','YieldFrom','Compare',
           'Call','FormattedValue','JoinedStr','Constant','Attribute',
           'Subscript','Starred','Name','List','Tuple','Slice']:
    _node_id(_n)

N_AST_NODE_TYPES = max(_NODE_COUNTER[0], 128)   # exported for embedding table size; padded for safety


# ── per-token feature container ───────────────────────────────────────────────

@dataclass
class TokenFeatures:
    node_types:   List[int] = field(default_factory=list)
    scope_depths: List[int] = field(default_factory=list)
    is_def:       List[int] = field(default_factory=list)
    is_call:      List[int] = field(default_factory=list)

    def __len__(self):
        return len(self.node_types)

    def pad_or_trim(self, length: int):
        """Ensure all lists have exactly *length* entries."""
        for attr in ('node_types', 'scope_depths', 'is_def', 'is_call'):
            lst = getattr(self, attr)
            if len(lst) < length:
                lst.extend([0] * (length - len(lst)))
            else:
                setattr(self, attr, lst[:length])


# ── line/col → token index mapping ───────────────────────────────────────────

def _build_offset_map(code: str) -> List[int]:
    """Return a list where offset_map[line] = char offset of line start (1-indexed lines)."""
    offsets = [0, 0]   # index 0 unused; index 1 = line 1 starts at char 0
    for i, ch in enumerate(code):
        if ch == '\n':
            offsets.append(i + 1)
    return offsets


def _char_offset(line: int, col: int, offset_map: List[int]) -> int:
    if line >= len(offset_map):
        return len(offset_map)
    return offset_map[line] + col


# ── main extractor ────────────────────────────────────────────────────────────

class ASTFeatureExtractor:
    """
    Extracts structural AST features aligned with the token sequence.

    For code that cannot be parsed (e.g. incomplete snippets) all features
    default to 0 so the model degrades gracefully.
    """

    def extract(self, code: str) -> TokenFeatures:
        # Step 1: tokenise to get the token sequence + positions
        raw_tokens = tokenize_python(code)
        n = len(raw_tokens)
        features = TokenFeatures(
            node_types   = [0] * n,
            scope_depths = [0] * n,
            is_def       = [0] * n,
            is_call      = [0] * n,
        )

        if n == 0:
            return features

        # Step 2: try to parse the AST
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return features   # return all-zero features for unparseable code

        offset_map = _build_offset_map(code)

        # Step 3: build a char-offset → (node_type, scope_depth, is_def, is_call) map
        char_map: Dict[int, Tuple[int, int, int, int]] = {}
        self._walk(tree, char_map, offset_map, scope_depth=0,
                   def_names=set(), call_names=set())

        # Step 4: align char-map with token positions
        # We reconstruct char offsets for each token by scanning through the code
        pos = 0
        for i, (tok_str, _) in enumerate(raw_tokens):
            # skip whitespace in the source to find where this token starts
            while pos < len(code) and code[pos] != tok_str[0]:
                pos += 1
            feat = char_map.get(pos, (0, 0, 0, 0))
            features.node_types[i]   = feat[0]
            features.scope_depths[i] = feat[1]
            features.is_def[i]       = feat[2]
            features.is_call[i]      = feat[3]
            pos += len(tok_str)

        return features

    # ── AST walker ────────────────────────────────────────────────────────────

    def _walk(self, node: ast.AST, char_map: dict, offset_map: List[int],
              scope_depth: int, def_names: set, call_names: set):
        node_type_id = _node_id(type(node).__name__)

        # Determine if this node increases scope depth
        is_scope = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef, ast.Lambda))
        child_depth = scope_depth + (1 if is_scope else 0)

        # Collect definition names at this node
        local_defs: set = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            local_defs.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    local_defs.add(t.id)
                elif isinstance(t, ast.Attribute):
                    local_defs.add(t.attr)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                local_defs.add(node.target.id)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            if isinstance(node.target, ast.Name):
                local_defs.add(node.target.id)
        elif isinstance(node, ast.arg):
            local_defs.add(node.arg)

        # Collect call names at this node
        local_calls: set = set()
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                local_calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                local_calls.add(node.func.attr)

        # Record char offset for Name nodes (most tokens map to these)
        if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
            offset = _char_offset(node.lineno, node.col_offset, offset_map)
            all_defs  = def_names  | local_defs
            all_calls = call_names | local_calls

            # is_def: this Name node's id is in the definition set
            is_d = 0
            if isinstance(node, ast.Name) and node.id in all_defs:
                is_d = 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef)):
                is_d = 1   # the name token right after def/class

            # is_call: this Name node is the function being called
            is_c = 0
            if isinstance(node, ast.Name) and node.id in all_calls:
                is_c = 1

            char_map[offset] = (node_type_id, scope_depth, is_d, is_c)

        # Recurse — pass accumulated defs/calls down
        for child in ast.iter_child_nodes(node):
            self._walk(child, char_map, offset_map, child_depth,
                       def_names | local_defs, call_names | local_calls)
