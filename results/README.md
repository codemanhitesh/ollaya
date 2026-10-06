# Results

Every measurement behind [ollaya.dev/results](https://ollaya.dev/results), as data. The site reads these files
at build time (`site/scripts/gen-results.mjs`), draws its charts from them and publishes each file unchanged
under `/data/results/`. Nothing on the results page is typed in by hand.

```text
results/
  machines.json            the machines the runs name, by id
  runs/*.json              one measurement session each
  runs/parity-docs.json    parity numbers compiled from docs/families/*.md, each with its citation
```

## Machines

`machines.json` maps an id to a machine. Ids are hardware names (`rtx-4090`, `rtx-5090`, `mac-m4-pro`, or
`community-<gpu>` for a contributor's machine), never host names.

| Field | Meaning |
|---|---|
| `label`, `headline` | how the page names it (`RTX 4090 desktop`, `RTX 4090`) |
| `kind` | `ours` (a test bench) or `community` (a contributor's report) |
| `cpu`, `gpus`, `ram_gb`, `os`, `driver` | the hardware and software, as the test bench section shows them |
| `uses` | what the machine measures, one line each |
| `by`, `source` | `community` only: who measured, and the link to their report |

## Runs

Every run file has `schema: 1`, a `suite`, a `date` and, except `parity-docs.json`, a `machine` id that
`machines.json` defines. The build fails on an unknown schema or machine. Three suites exist:

- **`latency-triage`**: the speed chart. One model at a time on one device, the five-question triage preset
  over HTTP to `/api/decide`, 30 different short messages in turn, 5 warm-up requests, then 20 timed
  (`protocol` records this). Each result has `model`, `p50_ms`, `p90_ms`, `min_ms`, `eval_p50_ms` (the
  runtime's own timing), `load_ms`, `first_request_ms`, `input_tokens`, `device` and `precision`.
  `convert/ollaya_convert/bench_latency.py` writes these files against a running `ollaya serve`; it needs only
  the Python standard library.
- **`public-benchmark`**: the accuracy and calibration charts. Bespoke Labs' public decision benchmark
  (3,880 questions) through `/v1/systemone`, scored by their own runner (`convert/ollaya_convert/bench_public.py`).
  Each result has `server`, `model`, `macro_accuracy`, `pooled_ece`, `brier`, `median_ms`, `p95_ms`, and
  `by_type` and `per_subset` breakdowns.
- **`parity`**: the runtime against the reference on one device (`max_logit_diff`, `max_prob_diff`,
  `decisions_same` of `questions`, `pass`), with an optional `cross_device` block for differences between
  devices, which are measured and not gated (docs/decisions/0003-llama-cpp-runtime.md, point 7).

## Adding a run

1. Measure on a machine in `machines.json`, or add the machine first (hardware name as the id).
2. Name the file after the session, as the others are (`2026-10-01-latency-rtx4090-cuda.json`); models
   added to an earlier session go in a new file with an `-extra` suffix, so the first stays as measured.
3. Keep the protocol of the suite. A different protocol is a different suite, not a new file in this one.
4. Check it: `cd site && npm run build` validates every file and redraws the page.

Never put host names, user names or tracker ids in these files: they are published as they are.
