Clef is Cloudflare's family of decision models, released under Apache-2.0. Clef-Flash is Qwen3.5-9B, fully post-trained, with a joint schema head: a small transformer that reads the model's hidden states and scores every option of every question together, in one forward pass per request. It never generates text, and its probabilities come straight from the head, with no fitted temperature.

> Needs Ollaya 0.9.0 or newer, the first release that runs the `clef-joint-v1` layout.

## Models

| Tag | Base | Params | Typed-decisions accuracy | Five questions, RTX 4090 |
|---|---|---|---|---|
| `clef:latest`, `clef:flash` | Qwen3.5-9B, post-trained | 9B | 0.703 | 532 ms |

Typed-decisions accuracy is the argmax against the majority label on all 400 typed-decisions states, measured by Ollaya, with a calibration error (ECE) of 0.020 and no fitted temperature. The time is the median of 15 warm requests of the triage preset through the HTTP API. Cloudflare reports Clef-Flash's results on their Decision Index on the [model card](https://huggingface.co/Cloudflare/clef-flash).

The larger Clef (Qwen3.8-27B, about 54 GB in BF16) is not in the library yet: it does not fit the GPUs Ollaya checks parity on.

## Usage

```shell
ollaya run clef --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `clef`.

## How it works

- **One sequence per request.** Ollaya builds the prompt exactly as Cloudflare's `encode_record` does: the state, then every question with its options as a schema. The head reads the hidden states over each question's instructions and each option, routes evidence from the state to every option, and scores all options of all questions at once.
- **Weights.** Cloudflare's four BF16 shards and the head's `joint_head.safetensors` download from Hugging Face, pinned to a commit and verified by sha256. Ollaya hosts only the ONNX graph (11 MB).
- **Parity.** Ollaya's Rust runtime matches Cloudflare's own code in fp32 on CUDA: identical token ids and spans, and the same decision on all 571 test questions, probabilities within 6.3e-6.

## Limits

- **Text only.** Clef reads images and video upstream; Ollaya runs its text path, so a request with images is rejected.
- **Questions.** Choice criteria must be an object of options, as upstream. Ollaya requires `instructions` on every question.
- **Length.** Up to 4,096 tokens per request, so a request fits next to the weights on a 24 GB GPU (upstream reads up to 16,384); a longer state is cut to fit.
- **Memory.** About 19 GB with the weights kept BF16: a 24 GB GPU.
