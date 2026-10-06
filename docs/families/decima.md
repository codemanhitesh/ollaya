# decima (`decima-late-interaction-v1`)

**Decima** by A. M. Madani ([amyrmahdy on Hugging Face](https://huggingface.co/amyrmahdy),
[amyrmahdy/decima](https://github.com/amyrmahdy/decima), Apache-2.0) is a family of multilingual encoders
fine-tuned with a small **late-interaction scorer**: the state and every option are encoded on their own, and
each option reads the state's tokens through cross-attention to get one score. Options never see each other, so
their order cannot change the answer. Score questions go through a cumulative-link ordinal head. Requested in
#54.

| Tag | Weights (fp32, `pytorch/` of the repository) | Encoder | Temperature |
|---|---|---|---|
| `decima:base`, `decima:latest` | `amyrmahdy/decima-base@2468005d` (Hub tag `v2.0`): `encoder/model.safetensors` (1.23 GB) and `head.safetensors` (57 MB) | `jhu-clsp/mmBERT-base` (MIT), 321M parameters | 0.7269 |
| `decima:agent` | `amyrmahdy/decima-agent@86a07aab` (Hub tag `v2.1`): the same two files, fine-tuned from decima-base for coding-agent decisions | the same | 0.6082 |
| `decima:small` | `amyrmahdy/decima-small@2e7f4d07` (Hub tag `v1.1.1`): `encoder/model.safetensors` (471 MB) and `head.safetensors` (19 MB) | `intfloat/multilingual-e5-small` (MIT), 122M parameters | 0.9356 |

The temperatures are the author's, from each checkpoint's `decima.json`. The code is pinned to the GitHub tag
`v1.1.1` (`2df60942`); the `decima/` package is unchanged at `v2.0.0` (`a58f99c9`), the release of decima-base
and decima-agent, and on `main` as of 2026-10-06. The author also publishes int8 ONNX exports; Ollaya's graphs
read the fp32 checkpoints instead, so that the weights come unmodified from the files the author's own code loads.

## Rows

`crates/ollaya-decision/src/decima.rs` ports `decima/systemone.py` (`to_question`) and the text and
tokenization steps of `decima/model.py`; `convert/ollaya_convert/families/decima/layout.py` is its Python twin.
Every question becomes one state row and one row per option, each encoded on its own:

```text
state row   [cls] tok(state_prefix  + normalize(text + "\n" + state))[..max_state_tokens - 2]  [sep]
option row  [cls] tok(option_prefix + normalize(text + " " + option))[..62]                    [sep]
```

- **Per model** (`decision.json`): small uses e5's prefixes `query: ` and `passage: `, `[cls]` 0 and `[sep]` 2;
  base and agent use no prefix, `<bos>` 2 and `<eos>` 1. `max_state_tokens` is 512, and 2,048 for agent.
- **Text.** The question's instructions: a string as is, anything else as `json.dumps(ensure_ascii=False)`,
  the question id when they are absent or null. A noul question whose criteria give a `true` or `false`
  description appends `\nTrue if: ...\nFalse if: ...`, with U+2014 for the side without one.
- **Options.** A choice's `label: description` (the label alone when the description is empty), each score
  level's text, and `yes` and `no` for noul.
- **State.** A string as is; an object or array as `json.dumps(ensure_ascii=False)`. Numbers, booleans and
  null are rejected, as upstream's server rejects them.
- **normalize.** Unicode NFC, then Python's `str.strip()`.
- **Length.** A state row keeps its first `max_state_tokens` tokens and is flagged: `/v1/systemone` answers 422
  `STATE_TRUNCATED`, as upstream's server does, and `/api/decide` answers from the cut state (as upstream's
  `model.py` does) with `state_truncated: true`. An option row is cut to 64 tokens silently, as upstream cuts
  it.

`python -m ollaya_convert.families.decima.check --model <slug>` compares the port with upstream on the shared
request set (the edge cases, the 400 typed-decisions rows and 9 inputs Decima's mapping treats specially). For
each model: 469 requests with identical rows (12 of them with a cut state for small, 7 for base, none for
agent), 18 rejected by both, 3 refused first by Ollaya's shared question rules, 0 differences.

## The graph

`python -m ollaya_convert.families.decima.export --model <slug>` writes one weightless graph that answers every
question of a request in one run. Its weight tensors (259 for small, 196 for base and agent) reference the two
checkpoint files by byte offset; only small's encoder pooler is left out.

```text
inputs   state_ids, state_mask     int64 [questions, state_len]   one state row per question, right-padded
         option_ids, option_mask   int64 [options, option_len]    every option row, questions in order
         option_state              int64 [options]                the question each option belongs to
outputs  scores                    float32 [options]              the raw score, before the temperature
         ordinal_g, ordinal_gap    float32 [options]              the ordinal head's two projections
```

The runner groups consecutive questions into one run as long as the padded tokens stay within the shared token
budget. Against upstream's PyTorch modules, on 295 questions, the scores are within 9.7e-6 (small), 1.7e-5
(base) and 1.3e-5 (agent), and the projections within 1.1e-5.

## Answers

- **Choice and noul.** A softmax of the scores over the temperature. Upstream's noul order is `[yes, no]`; the
  runner turns it into the `[false, true]` order answers use.
- **Score.** The ordinal head, on the scores over the same temperature, so the score slot of
  `calibration.json` is 1:

  ```text
  s = scores / T                    expected = sum_k softmax(s)_k * k - (K - 1) / 2
  g = mean_k ord_g(z_k) + expected  (ord_g is linear, so the graph returns it per option)
  gaps = softplus(ord_gap(z)) + 1e-3,  theta_j = sum_{i<=j} gaps_i - sum(gaps) / 2   (j < K - 1)
  p_k = sigmoid(g - theta_{k-1}) - sigmoid(g - theta_k)   (1 for k = 0, 0 for k = K - 1)
  log p = log_softmax(log(max(p, 1e-7)))
  ```

## Differences from upstream

- **Option cache.** Upstream's `Decima` caches the encodings of up to 256 option sets; Ollaya encodes every
  row on every request, in the same run as the state.
- **int8.** Not used (above).

## Parity (measured 2026-10-06)

Goldens: `python -m ollaya_convert.families.decima.goldens out/<slug>` runs the author's `model.py` and
`systemone.py` in fp32 on the CPU: 140 requests per model, 18 of them rejected upstream, 3 with a cut state for
small and base (none for agent, whose states hold 2,048 tokens). `parity_decima` checks the rejections and
every row id for id, then the graph's outputs, the decisions and the `/v1/systemone` answers against upstream's
`system_one`. On CPU the ONNX Runtime is pyke's 1.28.0, linked statically; on CUDA it is Microsoft's 1.28.2 from
the CUDA 13 pack.

| Model | Device | Questions | Rows | Decisions | Scores max | Probabilities max |
|---|---|---|---|---|---|---|
| small | x86-64 CPU | 581 | 2,922 identical | 581/581 | 1.1e-5 | 2.3e-6 |
| small | CUDA, RTX 4090 | 581 | 2,922 identical | 581/581 | 1.1e-5 | 1.3e-6 |
| small | CUDA, RTX 5090 | 581 | 2,922 identical | 581/581 | 1.2e-5 | 1.5e-6 |
| base | x86-64 CPU | 581 | 2,922 identical | 581/581 | 2.0e-5 | 3.0e-6 |
| base | CUDA, RTX 4090 | 581 | 2,922 identical | 581/581 | 9.1e-6 | 3.5e-6 |
| base | CUDA, RTX 5090 | 581 | 2,922 identical | 581/581 | 9.2e-6 | 2.7e-6 |
| agent | x86-64 CPU | 581 | 2,922 identical | 581/581 | 2.0e-5 | 4.3e-6 |
| agent | CUDA, RTX 4090 | 581 | 2,922 identical | 581/581 | 1.6e-5 | 5.5e-6 |
| agent | CUDA, RTX 5090 | 581 | 2,922 identical | 581/581 | 1.0e-5 | 2.8e-6 |

The ordinal projections are within 6.8e-5 everywhere. On every device the `/v1/systemone` answers name the same
choice as upstream's on every question, and their numbers agree to the fourth decimal (1.0e-4, the rounding).

## Quality

Typed-decisions: all 400 test states, 2,000 questions, argmax against the majority label, through Ollaya's
runtime (`logits` example and `quality_runtime.py`; small on the CPU, base and agent on CUDA). ECE at the
author's temperature.

| Model | Accuracy | Choice | Score | Noul | ECE |
|---|---|---|---|---|---|
| base | **0.495** (989 of 2,000) | 0.432 | 0.496 | 0.555 | 0.073 |
| agent | 0.486 (971) | 0.470 | 0.380 | 0.642 | 0.163 |
| small | 0.432 (863) | 0.412 | 0.356 | 0.552 | 0.110 |

The author reports 0.427 (95% interval 0.399 to 0.455) for small 1.1 and notes that every small model they
tested was below this benchmark's majority-class baseline of 0.461; base and agent are above it. The author
gives no typed-decisions figure for base or agent. Agent is trained for the decisions inside a coding agent's
loop (secret and command gates, tool, command and model choice); the author measures it on 130 hand-written
agent decisions (0.93, against 0.58 for base).

## Speed (measured 2026-10-06)

The triage preset (five questions) through the HTTP API, one request at a time (`bench_latency.py`, the
protocol of the 2026-10-01 sweep), on the RTX 4090 machine:

| Model | RTX 4090 p50 | p90 | CPU (i9-13900K) p50 | p90 | Load (CUDA / CPU) |
|---|---|---|---|---|---|
| small | 7.3 ms | 8.5 ms | 146 ms | 148 ms | 0.84 s / 0.58 s |
| base | 15.1 ms | 15.4 ms | 438 ms | 452 ms | 1.43 s / 1.37 s |
| agent | 14.8 ms | 15.9 ms | 439 ms | 442 ms | 1.27 s / 1.29 s |

On this CPU no other model Ollaya ships is as fast as small: the next are `laya:multilingual` at 308 ms and
`gliclass` at 563 ms (2026-10-01). The author measured one question with four options on one CPU core with
their int8 exports and cached option encodings (14 ms for small, 47 ms for base on an NVIDIA GB10's ARM cores);
Ollaya runs fp32 and encodes the options on every request.

## Limits

- **State.** 512 tokens per question for small and base, 2,048 for agent, the question's text included.
  Longer: `/v1/systemone` answers `STATE_TRUNCATED`, `/api/decide` answers from the first tokens.
- **Options.** Each option row holds the question's text and the option in 64 tokens, so long instructions cut
  the option's own text short, as upstream does. 2 to 255 choices, 2 to 10 score levels, 1 to 256 questions.
- **Languages.** Multilingual. For small the author evaluated 20 languages (the weakest are Swahili and Hindi);
  for base, Persian costs about 5 points against English on the same tasks, and other languages are less tested.
