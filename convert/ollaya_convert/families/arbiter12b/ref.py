"""The PyTorch reference for hiteshluke/arbiter-v4-12b in fp32: the reference prompt and the reference model.
The 12B sibling of `families/arbiter/ref.py`.

Arbiter v4 is a LoRA (r 32, alpha 64) on `google/gemma-4-12b-it` plus a 28-slot head,
`nn.Linear(3840, 28, bias=False)`, read at the last position of the prompt's final hidden state (after the
final norm). The model repository holds the adapter, `head.pt` (the EMA-averaged head, BF16) and
`head_meta.json`. Training saw score prompts of up to 10 digit levels (`0..9`); a score with any other number
of levels goes through the choice template with its rendered levels as the options.

    tok = ref.tokenizer(base)
    rows, meta = ref.encode(tok, state, questions)       # reference prompt rows
    model, head = ref.load(run, base, device="auto")
    scores = ref.forward(model, head, rows)               # one [28] array of raw slot scores per row
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np

from .layout import LayoutError, NUM_SLOTS, SCORE_LEVELS, VERBALIZERS, to_record

ARBITER_GIT = {"repo": "https://github.com/CodekinsTech/arbiter", "commit": "", "script": "training_v3_9_12b/kaggle_current.py"}
MODELS = {
    "arbiter-v4-12b": {
        "repo": "hiteshluke/arbiter-v4-12b",
        "revision": "c1cf6feb05e2f78bf33b1cc2370bf495d0b824c8",
        # Non-gated mirror of google/gemma-4-12b-it (gated=False), so the runtime pulls tokenless.
        # A single model.safetensors (bf16, ~23.9 GB) + tokenizer.json. TODO(verify): sha-match the text
        # tensors against google/gemma-4-12b-it before the PR, as the 4B doc does for its base.
        "base": "unsloth/gemma-4-12b-it",
        "base_revision": "55cdba0740a9765956f49501f689a66b098feda3",
        "base_files": ["model.safetensors"],
    },
}
MAX_ROW_TOKENS = 8192


class RequestError(ValueError):
    """The request is invalid for this model (HTTP 422)."""


# ---- training prompt templates (12B: 10-level score block) ----
def _make_choice_prompt(state, question, options):
    letters = [chr(ord('A') + i) for i in range(len(options))]
    opt_lines = '\n'.join(f'{L}. {o}' for L, o in zip(letters, options))
    return f'State: {state}\n\nQuestion: {question}\n\nOptions:\n{opt_lines}\n\nAnswer:'


def _make_noul_prompt(state, question):
    return (f'State: {state}\n\nQuestion: {question}\n\n'
            f'Options:\nT. Yes / True\nF. No / False\n\nAnswer:')


def _make_score_prompt(state, question):
    levels = '\n'.join(str(i) for i in range(SCORE_LEVELS))   # 0..9
    return f'State: {state}\n\nQuestion: {question}\n\nOptions:\n{levels}\n\nAnswer:'


NOUL_SLOTS = [0, 1]                               # T, F
def choice_slots(n): return list(range(2, 2 + n))
SCORE_SLOTS = list(range(18, 18 + SCORE_LEVELS))  # 18..27
# ----


def slots(qtype: str, k: int) -> List[int]:
    if qtype == "noul":
        return [NOUL_SLOTS[1], NOUL_SLOTS[0]]     # Ollaya order: false, true
    if qtype == "choice":
        return choice_slots(k)
    return list(SCORE_SLOTS)


def snapshot(slug: str):
    from huggingface_hub import snapshot_download
    m = MODELS[slug]
    run = os.environ.get("ARBITER_RUN") or snapshot_download(m["repo"], revision=m["revision"])
    if os.environ.get("ARBITER_BASE"):
        base = os.environ["ARBITER_BASE"]
    else:
        patterns = ["*.json", "*.jinja", "tokenizer.model", "tokenizer.json"] + (m["base_files"] or ["*.safetensors"])
        base = snapshot_download(m["base"], revision=m["base_revision"], allow_patterns=patterns)
    return run, base


def tokenizer(base_dir: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def encode(tok, state, questions):
    try:
        state_text, qs, meta = to_record(state, questions)
    except LayoutError as e:
        raise RequestError(str(e)) from e
    rows = []
    for (t, instructions, options), m in zip(qs, meta):
        if t == "noul":
            prompt = _make_noul_prompt(state_text, instructions)
        elif t == "choice" or options:   # a score of other than 10 levels is asked as a choice over its levels
            prompt = _make_choice_prompt(state_text, instructions, options)
        else:
            prompt = _make_score_prompt(state_text, instructions)
        ids = list(tok(prompt)["input_ids"])
        if len(ids) > MAX_ROW_TOKENS:
            raise RequestError("question %r: the row is %d tokens; this model reads up to %d"
                               % (m["qid"], len(ids), MAX_ROW_TOKENS))
        rows.append({"ids": ids, "last_pos": len(ids) - 1, "slots": slots("choice" if options else t, m["k"])})
    return rows, meta


def head_meta(run_dir: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, "head_meta.json")) as f:
        meta = json.load(f)
    if meta.get("num_slots") not in (None, NUM_SLOTS):
        raise SystemExit("head_meta.json: %d slots, expected %d" % (meta.get("num_slots", 0), NUM_SLOTS))
    return meta


def text_model(root):
    """The Gemma 4 text decoder inside the (peft or plain) model: the module with layers + rotary_emb + norm."""
    for mod in root.modules():
        if all(hasattr(mod, a) for a in ("layers", "rotary_emb", "embed_tokens", "norm")):
            return mod
    raise SystemExit("Gemma 4 text model not found (no module with layers/rotary_emb/embed_tokens/norm)")


def load(run_dir: str, base_dir: str, device: str = "cpu", merge: bool = True):
    """Gemma 4 12B IT + the adapter (peft) + the 28-slot head, all fp32. Returns (model, head)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    head_meta(run_dir)
    kw = {}
    if device == "auto":
        kw = {"device_map": "auto",
              "max_memory": {i: int(torch.cuda.get_device_properties(i).total_memory * 0.8)
                             for i in range(torch.cuda.device_count())}}
    base = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.float32,
                                                attn_implementation="eager", **kw)
    model = PeftModel.from_pretrained(base, run_dir)
    if merge:
        model = model.merge_and_unload()
    if device != "auto":
        model.to(device)
    model.eval()

    tm = text_model(base)
    head = torch.nn.Linear(tm.config.hidden_size, NUM_SLOTS, bias=False)   # 3840
    state = torch.load(os.path.join(run_dir, "head.pt"), map_location="cpu", weights_only=True)
    head.load_state_dict({"weight": state["proj.weight"].float()})
    head.to(tm.norm.weight.device if device == "auto" else device).eval()
    for p in list(model.parameters()) + list(head.parameters()):
        p.requires_grad_(False)
    return model, head


def forward(model, head, rows) -> List[np.ndarray]:
    """Raw 28-slot scores (float64), one unpadded row at a time."""
    import torch
    device = model.get_input_embeddings().weight.device
    hdev = head.weight.device
    out = []
    with torch.no_grad():
        for r in rows:
            o = model(input_ids=torch.tensor([r["ids"]], device=device), output_hidden_states=True,
                      use_cache=False)
            h = o.hidden_states[-1][0, -1]
            out.append(head(h.float().to(hdev)).double().cpu().numpy())
    return out
