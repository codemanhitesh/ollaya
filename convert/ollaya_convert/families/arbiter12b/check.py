"""Check the `arbiter-fixed-v2` port (layout.py, the Rust runtime's algorithm) against the reference prompt
(ref.py: the training script's templates, tokenized by transformers), id for id, without the model.

    uv run --no-project --with transformers==4.57.6 --with numpy \
        python -m ollaya_convert.families.arbiter12b.check <base snapshot> [--requests shared.jsonl] [--td 40]

<base snapshot> is `unsloth/gemma-4-12b-it` at the pinned revision (only its tokenizer files are read). The
requests are the shared set, a JSONL file of {"id", "state", "questions"} (`--requests`; else
`llm_common.cases`: the edge cases and `--td` typed-decisions rows), and `EXTRA` below. Every request: the
reference rows (`ref.encode`) against `Arbiter12bLayout` on the base's `tokenizer.json` with the decision.json
fields the export writes; rejections must match, and so must each question of a rejected request on its own.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import tokenizers

from . import ref
from .layout import Arbiter12bLayout, LayoutError, decision_fields


_UPSET = {"type": "score", "instructions": "How upset is the customer?",
          "criteria": ["calm", "mild", "annoyed", "upset", "angry", "furious"]}
_UPSET10 = {"type": "score", "instructions": "How upset is the customer, 0-9?",
            "criteria": [str(i) for i in range(10)]}  # 10 levels: the trained digit block
_INTENTS = {"type": "choice", "instructions": "Which intent best matches the message?",
            "criteria": {"intent_%02d" % i: "customer intent number %d" % i for i in range(16)}}
# The shared set has no 10-level score and no choice with 16 options, so these cover the head's edges:
# a 10-level score (the trained digit block), a 6-level score and scores of 2 and 16 levels (asked as a
# choice), and the two
# rejections of the fixed head (17 options, 17 levels); the long state crosses the 1,024-token sliding window.
EXTRA = [
    ("arbiter12b/score_10", "Third failed call today. Nobody listens and I am done waiting.", {"upset": _UPSET10}),
    ("arbiter/score_6", "Third failed call today. Nobody listens and I am done waiting.", {"upset": _UPSET}),
    ("arbiter/choice_16", "I want to change the shipping address of order A-104.", {"intent": _INTENTS}),
    ("arbiter/mixed", {"from": "user@acme.com", "body": "Billed twice for March, refund it today or we cancel.",
                       "amount": 49.5, "vip": True},
     {"refund": {"type": "noul", "instructions": "Is this a refund request?"},
      "team": {"type": "choice", "instructions": "Which team?",
               "criteria": {"billing": "charges and refunds", "tech": {"covers": ["bugs", "outages"]}, "other": None}},
      "upset": _UPSET}),
    ("arbiter/long_state", " ".join(["The checkout page fails intermittently after the latest deploy."] * 200),
     {"outage": {"type": "noul", "instructions": "Is this an outage?"}, "upset": _UPSET}),
    ("arbiter/choice_17_rejected", "I want to change my address.",
     {"ok": {"type": "noul", "instructions": "Is the message polite?"},
      "intent": {**_INTENTS, "criteria": {**_INTENTS["criteria"], "intent_16": "one more"}}}),
    ("arbiter/score_2_and_16", "Third failed call today.",
     {"ok": {"type": "noul", "instructions": "Is the message polite?"},
      "upset": {**_UPSET, "criteria": ["calm", "upset"]},
      "fine": {**_UPSET, "criteria": ["level %d of 16" % i for i in range(16)]}}),
    ("arbiter/score_17_rejected", "Third failed call today.",
     {"ok": {"type": "noul", "instructions": "Is the message polite?"},
      "upset": {**_UPSET, "criteria": ["level %d of 17" % i for i in range(17)]}}),
]


def requests(path=None, td=40):
    """The shared request set, (id, state, questions), from a JSONL file of {"id", "state", "questions"} or
    else `llm_common.cases`, followed by `EXTRA`."""
    if path:
        with open(path, encoding="utf-8") as f:
            shared = [(r["id"], r["state"], r["questions"]) for r in (json.loads(x) for x in f if x.strip())]
    else:
        from ..llm_common import cases

        shared = list(cases.edge_cases()) + list(cases.typed_decisions(td))
    return shared + EXTRA


def decision_for(tok):
    """decision.json's layout fields from the reference tokenizer (what export.py writes)."""
    return decision_fields(tok.bos_token_id, tok.pad_token_id, ref.MAX_ROW_TOKENS)


def port(base_dir, tok):
    return Arbiter12bLayout(tokenizers.Tokenizer.from_file(os.path.join(base_dir, "tokenizer.json")), decision_for(tok))


def compare(tok, lay, state, questions):
    """-> None when both sides agree, else a description of the first difference."""
    try:
        up, ue = ref.encode(tok, state, questions)[0], None
    except ref.RequestError as e:
        up, ue = None, e
    try:
        ours, oe = lay.encode(state, questions)[0], None
    except LayoutError as e:
        ours, oe = None, e
    if (ue is None) != (oe is None):
        return "rejection differs: reference %s, port %s" % (ue, oe)
    if ue is not None or up == ours:
        return None
    q = next(i for i, (x, y) in enumerate(zip(up, ours)) if x != y)
    u, o = up[q]["ids"], ours[q]["ids"]
    at = next((i for i, (x, y) in enumerate(zip(u, o)) if x != y), min(len(u), len(o)))
    return "question %d: %d vs %d ids, first difference at %d: %s vs %s" % (
        q, len(u), len(o), at, u[max(0, at - 3):at + 3], o[max(0, at - 3):at + 3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="local snapshot of unsloth/gemma-4-12b-it at the pinned revision")
    ap.add_argument("--requests", default=None, help="the shared request set as JSONL")
    ap.add_argument("--td", type=int, default=40, help="typed-decisions rows when --requests is not given")
    a = ap.parse_args()
    tok = ref.tokenizer(a.base)
    lay = port(a.base, tok)
    same = rejected = rows = 0
    bad = []
    for cid, state, qs in requests(a.requests, a.td):
        diff = compare(tok, lay, state, qs)
        if diff is None:
            try:
                rows += len(lay.encode(state, qs)[0])
                same += 1
            except LayoutError:
                rejected += 1
                # each question on its own, as goldens.py records the accepted ones
                for qid, q in qs.items() if isinstance(qs, dict) else []:
                    d = compare(tok, lay, state, {qid: q})
                    if d is not None:
                        bad.append(("%s %s" % (cid, qid), d))
                    else:
                        try:
                            rows += len(lay.encode(state, {qid: q})[0])
                        except LayoutError:
                            pass
        else:
            bad.append((cid, diff))
    print("identical requests %d | both reject %d | identical rows %d | mismatches %d" % (same, rejected, rows, len(bad)))
    for b in bad[:15]:
        print("  ", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
