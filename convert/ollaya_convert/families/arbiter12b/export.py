"""Export hiteshluke/arbiter-v4-12b (Gemma 4 12B IT + LoRA + 28-slot head) to a weightless ONNX graph.
The 12B sibling of `families/arbiter/export.py`.

    uv run python -m ollaya_convert.families.arbiter12b.export arbiter-v4-12b --out out/arbiter-v4-12b \
        [--run RUN --base BASE] [--device auto]

Graph (layout `arbiter-fixed-v2`, see layout.py and docs/families/arbiter12b.md):
    inputs   input_ids   int64   [rows, seq]  one row per question; seq a multiple of 64, right-padded
             last_pos    int64   [rows]       position of the row's last token (where the head is read)
    outputs  scores      float32 [rows, 28]   raw scores of the 28 head slots

The trunk is transformers' Gemma 4 text model recomputed by `llm_common/gemma4.py` (cache-free prefill,
causal + sliding masks, dual RoPE, KV-sharing tail; no mask input). The LoRA is NOT merged: every adapted
Linear runs as `x W^T + s * (x A^T) B^T`, so the graph keeps the upstream files byte-referenced: the base
`model.safetensors` (bf16, `unsloth/gemma-4-12b-it`), the adapter `adapter_model.safetensors` (F32) and the
head in `head.pt` (BF16 torch zip, uncompressed). The tokenizer is the base repository's `tokenizer.json`.

NOTE: the fp32 trace of a 12B is ~48 GB. Run with `--device auto` to split the model over the GPUs and CPU
(accelerate offload), or on a box with >=64 GB RAM. The weightless rewrite then references the on-disk bf16/
f32 files, so the shipped graph is tiny.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil

import torch

from ..llm_common import onnx_export as ox
from ..llm_common.gemma4 import Gemma4Trunk
from ...weightless_sharded import safetensors_source, torchzip_source
from . import ref
from .layout import CHOICE_LETTERS, NUM_SLOTS, SCORE_LEVELS, VERBALIZERS, decision_fields

INPUT_NAMES = ["input_ids", "last_pos"]
OUTPUT_NAMES = ["scores"]


class Arbiter12bGraph(torch.nn.Module):
    def __init__(self, text_model, head):
        super().__init__()
        self.trunk = Gemma4Trunk(text_model)
        self.head = head

    def forward(self, input_ids, last_pos):
        h = self.trunk(input_ids).float()
        rows = torch.arange(h.shape[0], device=h.device)
        return self.head(h[rows, last_pos.to(h.device)])


def rename(name):
    """Graph initializer name -> checkpoint tensor names (base model.safetensors, adapter or head.pt).

    Verified against the real checkpoints:
      base   model.safetensors  keys  `model.language_model.<x>.weight`
      adapter adapter_model...   keys  `base_model.model.model.language_model.<x>.lora_{A,B}.weight`
      head   head.pt            key   `proj.weight`
    """
    if name.startswith("trunk.m."):
        x = name[len("trunk.m."):]
        if ".lora_" in x:
            return ["base_model.model.model.language_model." + x.replace(".default", "")]
        return ["model.language_model." + x.replace(".base_layer", "")]
    if name.startswith("head."):
        return ["proj." + name[len("head."):]]
    return [name]


def export(slug, out_dir, run_dir, base_dir, device):
    meta = ref.MODELS[slug]
    tok = ref.tokenizer(base_dir)
    model, head = ref.load(run_dir, base_dir, device=device, merge=False)
    text_model = ref.text_model(model.base_model.model)   # the Gemma 4 text model (peft LoRA Linear wrappers)
    graph = Arbiter12bGraph(text_model, head).eval()

    # three rows (one per type, 64..192 tokens): every dynamic axis > 1
    rows, _ = ref.encode(tok, "The customer was charged twice for order A-104 and wants the duplicate refunded. " * 8,
                         {"a": {"type": "noul", "instructions": "Was the customer charged twice?"},
                          "b": {"type": "choice", "instructions": "Which team?",
                                "criteria": {"billing": "charges", "tech": "bugs", "other": None}},
                          "c": {"type": "score", "instructions": "How upset, 0-9?",
                                "criteria": list(range(10))}})   # 10 levels -> the trained digit block
    T = max(2, -(-max(len(r["ids"]) for r in rows) // ox.SEQ_MULTIPLE)) * ox.SEQ_MULTIPLE
    ids = torch.full((len(rows), T), tok.pad_token_id, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r["ids"])] = torch.tensor(r["ids"])
    args = (ids, torch.tensor([r["last_pos"] for r in rows]))
    with torch.no_grad():
        want = torch.tensor(ref.forward(model, head, rows), dtype=torch.float32)
        got = graph(*args).float().cpu()
    print("eager graph vs reference (unmerged LoRA, batched vs one row at a time): %.2e"
          % float((got - want).abs().max()))

    R = torch.export.Dim("rows", min=1, max=4096)
    N = torch.export.Dim("chunks", min=1, max=4096)
    dyn = {"input_ids": {0: R, 1: ox.SEQ_MULTIPLE * N}, "last_pos": {0: R}}
    tmp = ox.scratch_dir("arbiter12b-export-")
    secs = ox.export_graph(graph, args, INPUT_NAMES, OUTPUT_NAMES, dyn, os.path.join(tmp, "model.onnx"))
    print("exported in %.0fs" % secs)
    del graph, text_model, model, head, want, got
    gc.collect()

    base_ckpts = [os.path.join(base_dir, f) for f in meta["base_files"]]
    sources = [
        safetensors_source(f, p, repo=meta["base"], revision=meta["base_revision"], filename=f)
        for f, p in zip(meta["base_files"], base_ckpts)
    ] + [
        safetensors_source("adapter_model.safetensors", os.path.join(run_dir, "adapter_model.safetensors"),
                           repo=meta["repo"], revision=meta["revision"], filename="adapter_model.safetensors"),
        torchzip_source("head.pt", os.path.join(run_dir, "head.pt"), repo=meta["repo"], revision=meta["revision"],
                        filename="head.pt"),
    ]
    report = ox.weightless(tmp, out_dir, sources, rename)
    ox.cleanup(tmp)
    shutil.copy(os.path.join(base_dir, "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))

    cfg = json.load(open(os.path.join(run_dir, "adapter_config.json")))
    decision = {
        "engine": "onnx",
        "family": "arbiter12b",
        "layout": "arbiter-fixed-v2",
        "upstream": {"repo": meta["repo"], "revision": meta["revision"], "base": meta["base"],
                     "base_revision": meta["base_revision"], "code": ref.ARBITER_GIT,
                     "lora": {"r": cfg["r"], "alpha": cfg["lora_alpha"], "scaling": cfg["lora_alpha"] / cfg["r"],
                              "merged": False}},
        "contract": {
            "inputs": {
                "input_ids": {"dtype": "int64", "shape": ["rows", "seq"],
                              "note": "one row per question; seq a multiple of 64; right-pad with any id (pad)"},
                "last_pos": {"dtype": "int64", "shape": ["rows"], "note": "position of the row's last token"},
            },
            "outputs": {"scores": {"dtype": "float32", "shape": ["rows", NUM_SLOTS],
                                   "note": "raw scores of the head's 28 slots; a question's option logits are "
                                           "the scores at its slots"}},
            "seq_multiple": ox.SEQ_MULTIPLE,
            "positions": "0..seq-1, implicit",
            "attention": "causal (sliding layers: the last %d positions); no mask input" % text_model_window(base_dir),
        },
        **decision_fields(tok.bos_token_id, tok.pad_token_id, ref.MAX_ROW_TOKENS),
        "max_options": len(CHOICE_LETTERS),
        "score_levels": SCORE_LEVELS,
        "verbalizers": VERBALIZERS,
        "templates": {
            "row": "[bos] tok(\"State: {render(state)}\\n\\nQuestion: {render(instructions)}\\n\\nOptions:\\n{block}"
                   "\\n\\nAnswer:\"); the head is read at the last token",
            "noul_block": "T. Yes / True\nF. No / False",
            "choice_block": "{letter}. {option}, one line per option, letters A..P",
            "choice_option": "{name} | {name}: {render(description)}",
            "score_block": "0\n1\n...\n9 for exactly 10 levels; any other count (1..16): the choice block over "
                           "render(level), letters A.. in level order",
            "render": "None->'', scalars->Python str(), list->'- item' lines, dict->'key: value' lines, 2-space nesting",
            "add_special_tokens": False,
        },
        "option_logits": {"noul": "scores[row, [1, 0]] (false = F, true = T)",
                          "choice": "scores[row, 2 + j] for option j < 16",
                          "score": "scores[row, 18 + level] for 10 levels, else scores[row, 2 + level] for "
                                   "up to 16 levels"},
        "opset": ox.OPSET,
        "precision": "fp32 compute; base weights and head BF16, widened by Cast; adapter F32",
        "weights_in_memory": ox.weights_in_memory(report),
    }
    calibration = {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
                   "source": "no fitted temperature ships with the checkpoint (T = 1); fit per-type later"}
    files = {
        "model": slug,
        "layers": [
            {"role": "graph", "path": "model.onnx", "hosted_by": "ollaya",
             "bytes": os.path.getsize(os.path.join(out_dir, "model.onnx")),
             "sha256": ox.sha256_file(os.path.join(out_dir, "model.onnx"))},
            *[ox.file_entry("weights/base", meta["base"], meta["base_revision"], f, p, location=f)
              for f, p in zip(meta["base_files"], base_ckpts)],
            ox.file_entry("weights/adapter", meta["repo"], meta["revision"], "adapter_model.safetensors",
                          os.path.join(run_dir, "adapter_model.safetensors"), location="adapter_model.safetensors"),
            ox.file_entry("weights/head", meta["repo"], meta["revision"], "head.pt",
                          os.path.join(run_dir, "head.pt"), location="head.pt")
            | {"note": "torch zip; the head tensor is stored uncompressed and referenced by byte offset"},
            ox.file_entry("tokenizer", meta["base"], meta["base_revision"], "tokenizer.json",
                          os.path.join(base_dir, "tokenizer.json")),
            {"role": "decision", "path": "decision.json", "hosted_by": "ollaya"},
            {"role": "calibration", "path": "calibration.json", "hosted_by": "ollaya"},
        ],
        "weightless": {k: v for k, v in report.items() if k != "unused"},
        "unused_checkpoint_tensors": {k: len(v) for k, v in report["unused"].items()},
    }
    ox.write_json(os.path.join(out_dir, "decision.json"), decision)
    ox.write_json(os.path.join(out_dir, "calibration.json"), calibration)
    ox.write_json(os.path.join(out_dir, "files.json"), files)
    print(json.dumps(files["weightless"]["stats"]), "inline bytes", files["weightless"]["inline_bytes"],
          "graph MB %.1f" % (files["layers"][0]["bytes"] / 2**20), "unused", files["unused_checkpoint_tensors"])


def text_model_window(base_dir):
    cfg = json.load(open(os.path.join(base_dir, "config.json")))
    return cfg.get("text_config", cfg)["sliding_window"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(ref.MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default=None, help="local snapshot of the arbiter-v4-12b repo")
    ap.add_argument("--base", default=None, help="local snapshot of the base repo at the pinned revision")
    ap.add_argument("--device", default="cpu", help="'cpu', 'cuda', or 'auto' (split over GPUs + CPU for the 48 GB fp32 trace)")
    a = ap.parse_args()
    run, base = (a.run, a.base) if a.run and a.base else ref.snapshot(a.model)
    export(a.model, a.out, run, base, a.device)


if __name__ == "__main__":
    main()
