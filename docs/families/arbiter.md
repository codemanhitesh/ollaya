# arbiter (`arbiter-fixed-v1`)

**Arbiter v3.3** ([hiteshluke/arbiter-4b](https://huggingface.co/hiteshluke/arbiter-4b), by Codekins Pvt Ltd ·
Zyot Lab) is Gemma 3 4B IT with a LoRA (r 16, α 32, on q/k/v/o and gate/up/down) and a **fixed 24-slot head**,
`nn.Linear(2560, 24, bias=False)`, read at the last position of the prompt. The head's rows started as the base
LM head's rows for the 24 answer tokens and were trained with the LoRA; the checkpoint ships the average of the
last five evaluation heads (BF16, `head.pt`). One forward pass answers one question. Training code:
`training_v3_6/train.py` in [CodekinsTech/arbiter](https://github.com/CodekinsTech/arbiter) at `e1cb30fd`.

| Tag | Weights | Temperature |
|---|---|---|
| `arbiter:4b`, `arbiter:latest` | `unsloth/gemma-3-4b-it@bf46152c`: two BF16 shards and `tokenizer.json`; `hiteshluke/arbiter-4b@0c44271c`: `adapter_model.safetensors` (F32) and `head.pt` | 1.0 (none fitted) |

The head has three fixed slot ranges, so a question has at most 16 options (choice). A score of exactly 6 levels
uses the trained digit block and the score slots; a score of any other number of levels (1 to 16) is expressed as
a choice over the rendered level labels and reads the choice slots:

| Slots | Tokens | Read for |
|---|---|---|
| 0, 1 | `T`, `F` | noul: option logits `[F, T]` (Ollaya's order: false, true) |
| 2..17 | `A`..`P` | choice: option j at slot 2 + j |
| 18..23 | `0`..`5` | score: level j at slot 18 + j |

### Base repository

`unsloth/gemma-3-4b-it@bf46152c` is a copy of Google's `google/gemma-3-4b-it@093f9f38`. The Hub reports the same
LFS sha256 and size for every file the runtime reads:

| File | sha256 | Bytes |
|---|---|---|
| `model-00001-of-00002.safetensors` | `eb5fd5e97ddd07b56778733e9653c07312529cb00980a318fc3e1c4e3b5a8f1f` | 4,961,251,752 |
| `model-00002-of-00002.safetensors` | `fdde0e5aa5ced0fa203b3d50f4ab78168b7e3a3e08c6349f5cc9326666e1bb13` | 3,639,026,128 |
| `tokenizer.json` | `4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795` | 33,384,568 |

`model.safetensors.index.json` and `tokenizer.model` are identical too. Only metadata differs (`config.json`,
`generation_config.json`, `tokenizer_config.json`, `special_tokens_map.json`, the README, and the copy's extra
`chat_template.jinja`), and the runtime reads none of it. The manifest points at the copy because
`google/gemma-3-4b-it` is gated (the Hub lists it as `gated: manual`: each account must request access and accept
the Gemma terms before it can download), and Ollaya pulls without a Hugging Face token. It is also the name the
training script loads the base and the tokenizer by.

The tokenizer comes from the base repository, not from `hiteshluke/arbiter-4b`: that repository's `tokenizer.json`
(sha256 `b666c93e…`) is the same tokenizer with truncation to 255 tokens switched on.

## Sequence

`convert/ollaya_convert/families/arbiter/layout.py` (the port) and `crates/ollaya-decision/src/arbiter.rs`. One
causal row per question, the training prompt after Gemma's `<bos>`:

```text
ids = [<bos>] + tok("State: {render(state)}\n\nQuestion: {render(instructions)}\n\nOptions:\n{block}\n\nAnswer:")

block   noul    "T. Yes / True\nF. No / False"
        choice  "A. {option 0}\nB. {option 1}\n..."      one line per option, letters A..P
        score   "0\n1\n2\n3\n4\n5"                        exactly 6 levels: trained digit block, score slots
        score   "A. {render(level 0)}\nB. ..."            other level counts (1..16): choice template, choice slots
```

- `render`: `null` is `""`, a scalar is Python's `str()` (`True`, `1.0`, `1e-05`), a list is `- item` lines and an
  object `key: value` lines, two spaces per level. A choice option is `name`, or `name: render(description)`.
- The noul and score descriptions are not part of the prompt: the model was trained on the fixed blocks.
- The head reads the last position (`last_pos = len(ids) - 1`).
- **Rejected (422), never truncated:** a choice whose criteria are not an object, or hold more than 16 options
  (`TOO_MANY_OPTIONS`); a score with more than 16 levels; a row over 8,192 tokens. A request with one such
  question is rejected whole, like any invalid request.

The reference prompt (`ref.py`) is the training script's three prompt functions, copied verbatim, tokenized by
transformers as training tokenizes (`tok(prompt)`, which adds `<bos>`). The request -> text mapping (`render`,
option texts, validation) is Ollaya's, shared by both: the training data was plain text.

## Graph

| Tensor | dtype | Shape | |
|---|---|---|---|
| `input_ids` | int64 | `[rows, seq]` | one row per question, right-padded; `seq` a multiple of 64 |
| `last_pos` | int64 | `[rows]` | the row's last token |
| `scores` | float32 | `[rows, 24]` | the head's raw slot scores at `last_pos` |

`export.py`: transformers' Gemma 3 text model recomputed by `llm_common/gemma3.py` (positions `0..seq-1`, no mask
input: every layer is causal, and the sliding layers see the last 1,024 positions), the LoRA unmerged
(`x·Wᵀ + 2·(x·Aᵀ)·Bᵀ`), and the head. Weightless: the base shards, the adapter and the head tensor inside
`head.pt` are referenced by byte offset; nothing is re-hosted. The vision tower and its LoRA tensors are unused.
`weights_in_memory` is `bf16`, about 8 GB. The runner (`crates/ollaya-runner/src/arbiter.rs`) batches rows
shortest first under 8,192 padded tokens per `session.run` and returns the scores at each question's slots.

## Differences from upstream

- **Precision.** Training and the model-card benchmarks ran the base 4-bit (NF4, bitsandbytes) with BF16 compute;
  the adapter was trained against that 4-bit base (`unsloth/gemma-3-4b-it-unsloth-bnb-4bit`). Ollaya runs the
  unquantized BF16 base in fp32, as does the reference here. Measured on the same rows, the unquantized base scores
  equal or higher (see Quality).
- **Length.** Training cut prompts at 768 tokens and the benchmark script at 1,024 (from the right, which drops
  `Answer:`). Ollaya reads rows of up to 8,192 tokens whole and rejects longer ones.
- **Requests.** Training saw text; JSON states and instructions go through `render` above. Gemma's control tokens
  written in user text (`<start_of_turn>`, `<bos>`) are tokenized as control tokens, as in training.
- **Fixed head.** Choices over 16 options and scores with more than 16 levels are rejected, not truncated or
  rescaled. Scores with 1..16 levels other than 6 are expressed as choices over the rendered level labels.
- **Temperature.** The checkpoint has no fitted temperature; `calibration.json` ships 1.0 for every type.

## Parity

The shared request set (the edge cases and 40 typed-decisions rows, as JSONL), plus arbiter-specific cases in
`check.py` (`EXTRA`): the shared set has no 6-level score and no 16-option choice, so these add both, a 2-level
and a 16-level score expressed as choices, a long state past the sliding window, and the rejection of a score
with more than 16 levels.

- **Prompt** (`check.py`, no model needed): the port against the reference prompt, id for id, on the base's
  tokenizer. Measured 2026-10-05 with transformers 4.57.6: 128 requests, 107 accepted and identical, 21
  rejected by both, 589 rows identical (every question of a rejected request is also compared on its own), 0
  mismatches.
- **Goldens** (`goldens.py`): the reference model in fp32 (LoRA merged, TF32 off, one unpadded row at a time):
  token ids, last position and slots, all 24 slot scores, option logits and probabilities per question.
- **Export** (`parity.py`): ONNX Runtime (LoRA unmerged, rows batched) against the reference.
- **Runtime** (`crates/ollaya-runner/examples/parity_arbiter.rs`): identical rows and rejections, every decision
  the same, all 24 slot scores within 1e-3.

```sh
cd convert
uv run --no-project --with transformers==4.57.6 --with numpy \
    python -m ollaya_convert.families.arbiter.check BASE --requests shared.jsonl
uv run --with peft==0.19.1 --with transformers==4.57.6 \
    python -m ollaya_convert.families.arbiter.export arbiter-4b --out out/arbiter-4b
uv run --with peft==0.19.1 --with transformers==4.57.6 \
    python -m ollaya_convert.families.arbiter.goldens out/arbiter-4b --requests shared.jsonl --device auto
uv run --with peft==0.19.1 --with transformers==4.57.6 \
    python -m ollaya_convert.families.arbiter.parity out/arbiter-4b --requests shared.jsonl --device auto
cd .. && cargo run --release -p ollaya-runner --example parity_arbiter -- \
    convert/out/arbiter-4b convert/out/goldens-arbiter-4b.jsonl cuda
```

`--device auto` splits the fp32 reference (about 17 GB with the unused vision tower) over every visible GPU; on
one GPU with enough memory, `--device cuda` does the same work.

**Measured 2026-10-05** on Kaggle (2x NVIDIA T4 16 GB, 4 vCPUs, 31 GB RAM, Ubuntu 24.04, driver 580), with
the weights at the pinned revisions (the base shards, `tokenizer.json`, `adapter_model.safetensors` and
`head.pt` downloaded from the Hub; the base sha256 values match the table above):

- **Prompt** (`check.py`): 128 requests, 107 identical, 21 rejected by both, 589 rows identical, 0 mismatches.
- **Export**: 126 s to trace, 361 s in all on the CPU; peak process RSS 20.3 GiB. The weightless `model.onnx`
  is 10.3 MB (921 external tensors, 445 casts); 439 tensors of the first base shard (the vision tower) and 162
  adapter tensors (its vision LoRA) are unused. The eager graph (LoRA unmerged, batched) is within 7.6e-6 of
  the reference on the export's three sample rows.
- **Goldens**: 146 records (21 rejected requests, each followed by its `#valid` questions), 589 questions,
  fp32 reference split over both T4s.
- **Export parity** (`parity.py`, ONNX Runtime 1.30 on the CPU): the v3 run was cancelled at 90 of 128
  requests (422 questions processed, no mismatches at that point). An earlier run on the pre-score-as-choice
  rows (420 questions) completed with 0 mismatches, max slot diff 7.2e-5 (noul), 5.3e-5 (score), 3.2e-5
  (choice), max probability difference 1.4e-5.
- **Runtime parity** (`parity_arbiter`, CUDA execution provider on one T4: Microsoft ONNX Runtime 1.28.2, CUDA
  12 build, loaded with `--features ollaya-runner/cuda-dynamic`; `weights_in_memory` bf16): 589 rows identical,
  0 row mismatches, 0 rejection mismatches, 0 questions refused by the shared question rules, the same
  decision on 589 of 589 questions. Largest slot score difference 1.7e-4 (tolerance 1e-3), probability
  difference max 1.5e-5, p99 9.6e-6. Requests of 3 or more questions (n = 118): p50 1,613 ms, p95 4,965 ms;
  GPU memory peak 9.7 GiB (`nvidia-smi`). The CPU execution provider was not run.

## Quality

- **Typed-decisions** (all 400 test states, argmax against the majority label, temperature 1, from the fp32
  reference: `python -m ollaya_convert.families.llm_common.eval_refs arbiter arbiter-4b out/arbiter-4b
  --device auto`, measured 2026-10-05 on 2x T4):

  | Type | Questions | Answered | Accuracy |
  |---|---|---|---|
  | noul | 600 | 600 | 0.677 |
  | choice | 600 | 600 | 0.563 |
  | score | 800 | 800 | 0.530 |
  | all | 2,000 | 2,000 (coverage 1.0) | 0.584 |

  The fixed head answers a score of exactly 6 levels with the trained digit block, and a score of any other
  number of levels as a choice over the rendered level labels (the same choice template with `render(level)` as
  option text). The typed-decisions scores have 4 or 5 levels, so all 800 are answered via the choice framing.
  Score accuracy (0.530) is above the majority-class baseline (0.338) but below noul and choice, as the model
  was not trained on this framing. Fitting one temperature per type on half the rows does not change the overall
  number materially (0.587 and 0.581 on the other half).
- **Published benchmarks** (measured on an NVIDIA T4 with the training prompt, one row at a time, not through
  Ollaya). The model card numbers come from the 4-bit base the adapter was trained on. The same adapter and head
  were then run on the same rows with the unquantized base (fp32 compute, the way Ollaya runs it):

| Benchmark | Type | n | 4-bit base (model card) | Unquantized base |
|---|---|---|---|---|
| BoolQ (validation) | noul | 1,000 | 0.849 | 0.853 |
| ARC-Challenge (test) | 4-way choice | 500 | 0.738 | 0.762 |
| CommonsenseQA (validation) | 5-way choice | 500 | 0.706 | 0.720 |
| OpenBookQA (test) | 4-way choice | 500 | 0.722 | 0.748 |

  The unquantized base is equal or better on all four (per-benchmark differences are within sampling noise, all in
  the same direction), so running the full BF16 base in Ollaya does not cost accuracy relative to the model card.

## Limits

- **Options.** Choices of 1..16 options; scores of exactly 6 levels use the trained digit block, other level
  counts (1..16) are expressed as choices.
- **Rows.** Up to 8,192 tokens per question, the whole state included; the model was trained on up to 768.
- **Memory.** About 8 GB of weights kept BF16; the CUDA parity run peaked at 9.7 GiB on a T4.

## License and attribution

- **Arbiter** (the LoRA adapter and the 24-slot head): Apache-2.0, by Codekins Pvt Ltd · Zyot Lab.
- **Base model:** Gemma 3 4B IT by Google DeepMind (https://huggingface.co/google/gemma-3-4b-it), under the
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and the
  [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). Ollaya fetches the weights
  unmodified and does not redistribute them.
- The catalog's `license` field says both: "Apache-2.0 (LoRA adapter and head) and the Gemma Terms of Use (Gemma 3
  base model)", and the license layer carries the Gemma notice.
- **Training data:** mainly [SargeDev/jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3);
  see the dataset card for its license.
