# Add the `arbiter` family (hiteshluke/arbiter-4b)

This PR submits **Arbiter**, Codekins Pvt Ltd / Zyot Lab's 4B decision model, as a new Ollaya family.

## What Arbiter is

- Base: `unsloth/gemma-3-4b-it` (Gemma 3 4B IT text path, hidden size 2560).
- LoRA adapter: r=16, α=32, dropout=0.05, targeting q/k/v/o + gate/up/down.
- **Fixed 24-slot pointer head**: `nn.Linear(2560, 24)` read at the final-position hidden state of
  the prompt. The 24 rows are initialized from the base LM-head rows for the verbalizer tokens
  `T, F, A..P, 0..5` and jointly trained with the LoRA; the checkpoint ships the EMA average of the
  last 5 eval heads.
- Prompt format identical to training: `State: / Question: / Options: / Answer:`.
- Three decision primitives in one forward pass per question:
  - `noul` → slots 0..1 (`T`, `F`),
  - `choice` → slots 2..17 (`A..P`, up to 16 options),
  - `score` → slots 18..23 (`0..5`, always 6 levels).

The model repository is `hiteshluke/arbiter-4b` (adapter + `head.pt` + `tokenizer.json`); the base
weights come unmodified from `unsloth/gemma-3-4b-it`.

## Why it fits the Ollaya conventions cleanly

- The runtime surface is one weightless ONNX graph (`[rows, seq]` ids + `[rows]` last-token index →
  `[rows, 24]` scores) plus a decision/calibration JSON pair. No new engine, no runtime Python.
- The LoRA stays unmerged in the graph, so base shards from `unsloth/gemma-3-4b-it` and the adapter
  from the Arbiter repo are both byte-referenced in the weightless export (nothing re-hosted).
- `head.pt` is a torch zip whose two tensors are stored uncompressed, so they can be referenced by
  byte offset — the same pattern Ollaya already uses for pointer heads.

## What this PR adds

- `convert/ollaya_convert/families/arbiter/` — `__init__.py`, `layout.py` (`arbiter-fixed-v1`),
  inline `ref.py` (Gemma 3 + LoRA + 24-slot head, no external `arbiter` package),
  `export.py` (weightless ONNX export + `decision.json` + `calibration.json`),
  `goldens.py` (5 primitive fixtures) and `parity.py` (reference vs ONNX, tolerance 1e-4).
- `convert/ollaya_convert/catalog.py` — one entry `"arbiter"` with tag `4b` and alias
  `latest -> 4b`; author `Codekins Pvt Ltd · Zyot Lab`, license Apache-2.0.
- `docs/families/arbiter.md` — architecture, prompt format, slot layout, benchmarks and licensing.
- `registry/v2/library/arbiter/manifests/{4b,latest}` — manifest shells, URLs pointed at
  `unsloth/gemma-3-4b-it` and `hiteshluke/arbiter-4b` on Hugging Face.

## Measured accuracy (ship time)

| Benchmark | Accuracy | n |
|---|---|---|
| BoolQ | 0.849 | 1,000 |
| ARC-Challenge | 0.738 | 500 |
| CommonsenseQA | 0.706 | 500 |
| OpenBookQA | 0.722 | 500 |

## What we tested locally

- `layout.ArbiterLayout.encode` reproduces the training prompt byte-for-byte for all three
  primitives (noul, 1..16-option choice, 6-level score).
- `ref.load` + `ref.forward` on the local checkpoint returns a `[rows, 24]` slot-score matrix with the
  expected argmax on each primitive golden.
- `export.ArbiterGraph` matches `ref.forward` (unmerged vs merged LoRA) to the usual ≈1e-6 on the
  slot-space scores.

## What maintainers need to finish

1. **Pin the Hugging Face revisions.** `ref.MODELS["arbiter-4b"]` and the manifests contain
   `PINNED_AT_PR_TIME` placeholders for the `hiteshluke/arbiter-4b` and `unsloth/gemma-3-4b-it`
   revisions. Pin them to the SHAs you resolve at merge time (we are happy to re-roll).
2. **Regenerate sha256s in the manifest.** `registry/v2/library/arbiter/manifests/{4b,latest}` ship
   with `sha256:ARBITER_*_PLACEHOLDER` digests; these are filled in by running the weightless export
   against the pinned revisions and hashing each HF file + the derived blobs. Running
   `python -m ollaya_convert.families.arbiter.export arbiter-4b --out out/arbiter-4b` produces a
   `files.json` with every real sha256.
3. **Parity gate.** Run `python -m ollaya_convert.families.arbiter.parity arbiter-4b
   out/arbiter-4b --onnx`; the shipped tolerance is 1e-4 on the 24-slot scores.

## Licensing and attribution

- Arbiter adapter + head: Apache-2.0.
- Base weights: Gemma 3 4B IT by Google DeepMind, under the Gemma Terms of Use.
- Training data: `SargeDev/jev-distill-corpus-v3` (see the dataset card for its own licensing).
