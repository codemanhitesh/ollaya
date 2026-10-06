Decima by [A. M. Madani](https://huggingface.co/amyrmahdy) is a family of multilingual decision models built on a late-interaction scorer. The state and each option are encoded on their own, and every option reads the state to get its score, so the order of the options never changes the answer. Score questions go through an ordinal head. It never generates text.

> Needs Ollaya 0.11.0 or newer, the first release that runs the `decima-late-interaction-v1` layout.

## Models

| Tag | Model | Typed-decisions accuracy | Five questions, RTX 4090 / CPU |
|---|---|---|---|
| `decima:latest`, `decima:base` | Decima-base 2.0 (mmBERT-base, 321M), fp32, 1.3 GB | 0.495 | 15 ms / 438 ms |
| `decima:agent` | Decima-agent 2.1 (decima-base for coding agents), fp32, 1.3 GB | 0.486 | 15 ms / 439 ms |
| `decima:small` | Decima-small 1.1 (multilingual-e5-small, 122M), fp32, 490 MB | 0.432 | 7.3 ms / 146 ms |

Typed-decisions accuracy is the argmax against the majority label on all 400 typed-decisions states, measured by Ollaya. Calibration error (ECE) at the author's temperatures: 0.073 for base, 0.163 for agent, 0.110 for small. The author reports 0.427 for small, and notes that small models struggle with this benchmark's long, multi-fact business cases. Latency is the median request through the HTTP API on an RTX 4090 and on the CPU of the same machine (i9-13900K).

`decima:small` is the fastest model Ollaya runs on a CPU. `decima:agent` is trained for the small, frequent decisions inside a coding agent's loop: is there a real secret in this edit, should this command run, ask or be denied, which tool or command fits, which model tier a task needs. The author measures 0.93 on 130 hand-written agent decisions, against 0.58 for base.

## Usage

```shell
ollaya run decima --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `decima`, `decima:agent` or `decima:small`.

## How it works

- **Rows.** Each question becomes one state row (the question with the state) and one row per option (the question with the option), exactly as the author's `decima/systemone.py` and `model.py` build them.
- **Scores.** The options of every question are scored in one pass. Choice and yes/no answers are a softmax of the scores at the author's fitted temperature; score questions use the author's cumulative-link ordinal head.
- **Engine.** ONNX Runtime, fp32, on an NVIDIA GPU (CUDA) or the CPU. The graph reads the author's own checkpoint files, unmodified.
- **Parity.** Ollaya's runtime matches the author's own code (fp32) on 581 questions per model, on the CPU and on CUDA (RTX 4090 and RTX 5090): identical token rows, the same decision on every question, probabilities within 5.5e-6.

## Limits

- **State.** 512 tokens per question (2,048 for `decima:agent`), the question included. `/v1/systemone` answers a longer state with `STATE_TRUNCATED`, as the author's server does; `/api/decide` answers from its first tokens.
- **Options.** Each option row holds the question and the option in 64 tokens, so a long question cuts the option's own text short, as upstream does. 2 to 255 choices, 2 to 10 score levels.
- **Languages.** Multilingual. The author evaluated small in 20 languages (the weakest are Swahili and Hindi). For base, the author reports Persian about 5 points below English on the same tasks, and other languages less tested.
