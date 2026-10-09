# arbiter12b (`arbiter-fixed-v2`) — PORT PLAN (work in progress)

**Arbiter v4 12B** (Codekins Pvt Ltd · Zyot Lab) is **Gemma 4 12B IT** with a LoRA (r 32, α 64, on
q/k/v/o and gate/up/down) and a **fixed 28-slot head**, `nn.Linear(3840, 28, bias=False)`, read at the last
position of the prompt — the 12B sibling of the merged [`arbiter`](arbiter.md) (4B, 24-slot) family. The head's
rows started as the base LM-head rows for the 28 answer tokens and were trained with the LoRA; the checkpoint
ships the average of the last evaluation heads (BF16, `head.pt`). One forward pass answers one question.

This file is the **port plan**, not a finished family doc. The Python layout is landed
(`convert/ollaya_convert/families/arbiter12b/layout.py`, 28 slots, validated). The remaining pieces, in
dependency order, are below.

## Head / slots (settled)

| Slots | Tokens | Read for |
|---|---|---|
| 0, 1 | `T`, `F` | noul: option logits `[F, T]` (Ollaya's order: false, true) |
| 2..17 | `A`..`P` | choice: option j at slot 2 + j (≤ 16 options) |
| 18..27 | `0`..`9` | score: level j at slot 18 + j (exactly 10 levels) |

Verbalizer `T F A..P 0..9` = 28 slots. A score of any count other than 10 is asked as a **choice** over its
rendered levels and reads the choice slots — same rule as the 4B (`arbiter`), so the Rust decision path is a
constant bump, not new logic. (Open parity question below.)

## Base model (Gemma 4 — the hard dependency)

`google/gemma-4-12b-it` is **architecturally new** vs Gemma 3, so the ONNX head path needs a **new recompute
module** `convert/ollaya_convert/families/llm_common/gemma4.py`. `gemma3.py` / `qwen3.py` do **not** cover it.
From the model config:

- `model_type: gemma4_unified_text`, `Gemma4UnifiedForConditionalGeneration`; 48 layers; hidden 3840.
- **Split heads by layer type:** sliding layers `head_dim 256`, 8 KV heads; full-attention layers
  `global_head_dim 512`, **1** KV head, **K = V shared** (`attention_k_eq_v: true`).
- `layer_types`: 40 sliding + 8 full, full every 6th layer (6, 12, …, 48); sliding window 1024.
- **Dual RoPE:** full layers `rope_theta 1e6`, `partial_rotary_factor 0.25`, type `proportional`; sliding layers
  `rope_theta 1e4`, type `default`.
- `final_logit_softcapping 30.0`, `rms_norm_eps 1e-6`, `tie_word_embeddings true`, vocab 262144.
- `query_pre_attn_scalar` / qk-norm / pre-post-FFN norms: **not in config — confirm against the transformers
  `Gemma4Unified` modeling code before writing the recompute.** This is the single largest task; it must match
  HuggingFace within the parity tolerance (every decision identical, every option score within 0.001).

Base is gated on the Hub (Ollaya pulls tokenless), so the manifest must point at a **non-gated mirror** pinned
by commit + sha256 (as the 4B points at `unsloth/gemma-3-4b-it@…`). Find/confirm an `unsloth/gemma-4-12b-it`
(or equivalent) mirror whose read files match `google/gemma-4-12b-it` by LFS sha256.

**Status: `gemma4.py` recompute is written and validated bit-exact** against transformers
`Gemma4UnifiedTextModel.forward` (max abs diff 0.0 on a tiny random Gemma-4 that exercises sliding + full
attention, the KV-sharing tail, dual partial RoPE, `v_norm`, and the per-layer `layer_scalar`). It reuses the
HF submodules and re-expresses only the cache-free additive masks, so it matches by construction. The dominant
risk of this port is retired.

## Weights (author's HF repo — never re-hosted)

Invariant (CLAUDE.md): Ollaya never re-hosts weights; it reads the author's repo unmodified at a pinned commit.

**Done.** Clean public repo [`hiteshluke/arbiter-v4-12b`](https://huggingface.co/hiteshluke/arbiter-v4-12b)
published at pinned commit **`c1cf6feb05e2f78bf33b1cc2370bf495d0b824c8`** (step-2500 adapter + ema-final head,
the 0.640 checkpoint):

| File | sha256 | Bytes |
|---|---|---|
| `adapter_model.safetensors` (F32 LoRA) | `88f16d44292b3ced…` | 524,648,560 |
| `head.pt` (28-slot, BF16) | `8bf9f80399aeb479…` | ~215 KB |
| `head_meta.json` (verbalizer, calibration_temp) | `45f065c76ecc1002…` | — |

(Full sha256s: read them from the repo's LFS pointers at export time.) Still TODO: a **non-gated Gemma-4 base
mirror** pinned by commit + sha for the manifest (base weights, like the 4B's `unsloth/gemma-3-4b-it@…`).

## Files to add (mirrors the merged `arbiter` family)

- [x] `convert/ollaya_convert/families/arbiter12b/layout.py` — 28-slot layout (landed, validated).
- [x] `convert/ollaya_convert/families/arbiter12b/__init__.py`.
- [x] `convert/ollaya_convert/families/llm_common/gemma4.py` — **Gemma 4 forward recompute** (the big one).
      **Landed + validated bit-exact** (max abs diff 0.0) vs transformers' own `Gemma4UnifiedTextModel.forward`
      on a tiny random model exercising sliding + full attention, the KV-sharing tail, dual RoPE, `v_norm` and
      `layer_scalar`. It reuses the HF decoder layers/rotary/norm and only re-expresses the cache-free masks.
- [ ] `convert/ollaya_convert/families/arbiter12b/{export.py, ref.py, goldens.py, parity.py, check.py}` — copy
      from `arbiter/`, swap `gemma3` → `gemma4`, 24 → 28, 6 → 10 score levels, point at the 12B weights.
- [ ] `crates/ollaya-decision/src/arbiter12b.rs` — copy `arbiter.rs`; `SCORE_LEVELS 6 → 10`,
      `SCORE_BLOCK "0..9"`, `num_slots 28`; register in `decision.rs` / `lib.rs`.
- [ ] `crates/ollaya-runner/src/arbiter12b.rs` (+ `examples/parity_arbiter12b.rs`) — copy `arbiter.rs` runner;
      the ONNX I/O is identical shape-wise (`input_ids`, `last_pos` → `scores[rows, 28]`).
- [ ] `convert/ollaya_convert/catalog.py` — register the `arbiter12b` family + `arbiter:12b` tag + manifest
      (base mirror @commit, author repo @commit, temperature).
- [ ] `docs/families/arbiter12b.md` — replace this plan with the finished family doc (base-repo sha table,
      sequence, graph, parity numbers), in `arbiter.md`'s format.

## Parity (the gate)

`cargo run --release -p ollaya-runner --example parity_arbiter12b -- <model-dir> <goldens.jsonl>` must pass:
every decision identical and every option score within **0.001** vs the fp32 transformers reference
(`ref.py` on GPU). Never loosen the tolerance. CUDA parity first (fast signal), then CPU.

**Open parity question (score reading).** Our head benchmark (0.640) read a *k*-level score (k = 2..10) off the
digit slots `18..18+k`. The ported family instead uses the digit block only for exactly-10-level scores and
asks every other count as a *choice* (matching `arbiter.rs`). The benchmark's score questions are mostly
4-level, so the two conventions read different slots → the Ollaya typed-decisions number for the 12B may differ
from 0.640. Decide at parity time: (a) keep the 4B convention (score-as-choice, simpler, consistent) and report
whatever it scores, or (b) add a flexible-digit path in `arbiter12b.rs` to reproduce the 0.640 reading. This
only affects the `score` type; noul/choice are unaffected.

## Sequence to finish

1. Publish clean 12B weights (tagged commit) + confirm a non-gated Gemma 4 base mirror; fill the sha tables.
2. Write + unit-check `gemma4.py` against transformers on a few prompts (the hard part).
3. Copy the arbiter export/ref/parity + Rust decision/runner; bump 24→28 / 6→10 / gemma3→gemma4.
4. Export the ONNX, run CUDA then CPU parity to 0.001, settle the score-reading convention.
5. Finished family doc; PR `codemanhitesh/ollaya` → `ollaya-dev/ollaya` (as the 4B PR #49 was done).

## Prior art

The 4B [`arbiter`](arbiter.md) family (PR #49, merged 2026-10-06) is the exact template. This is the same port
with a newer base and a 4-slot-wider score block.
