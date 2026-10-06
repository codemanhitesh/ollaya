"""Check the `clef-joint-v1` port (layout.py) against upstream `encode_record`, id for id, without the model.

    uv run python -m ollaya_convert.families.clef.check <model snapshot> [--td 200]

Every case of the shared set: upstream's systemone checks + `encode_record` on transformers' tokenizer (what
upstream runs) against `ClefLayout` on the repository's tokenizer.json loaded by HF `tokenizers` (what the
runtime runs), with `decision_for` below. Ids, question spans, option spans and option ids must be identical
and rejections must match. Requests Ollaya's shared question parser rejects first are counted on their own.
"""
from __future__ import annotations

import argparse
import os
import sys

import tokenizers

from ..llm_common import cases
from . import ref
from .layout import ClefLayout, LayoutError, ollaya_rule

# encode_record's fixed strings (joint_schema_model.py at ref.REVISION); the check proves them against upstream.
SCHEMA = "\n\nSCHEMA FIELDS:\n"
FIELD = "\nFIELD {n}\nID: {id}\nTYPE: {type}\nINSTRUCTION: "
OPTIONS = "\nALLOWED OPTIONS:\n"
OPTION = "OPTION {n}: "
OPTION_END = "\n"
FIELD_END = "END FIELD\n"
USER = "<|im_start|>user\nSTATE:\n"
SUFFIX = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"


def decision_for(jsm):
    """The layout's fields of decision.json, from the upstream module."""
    noul = dict(jsm.question_options({"type": "noul"}))
    return {
        "prompt": {"prefix": "<|im_start|>system\n%s<|im_end|>\n%s" % (jsm.SYSTEM_PROMPT, USER),
                   "schema": SCHEMA, "field": FIELD, "options": OPTIONS, "option": OPTION,
                   "option_end": OPTION_END, "field_end": FIELD_END, "suffix": SUFFIX},
        "noul_criteria": {"true": noul["true"], "false": noul["false"]},
        "max_tokens": ref.MAX_LEN,
    }


def upstream_row(enc):
    """encode_record's EncodedRecord in the port's shape."""
    return {"ids": list(enc.input_ids),
            "questions": [{"qid": q.question_id, "span": list(q.question_span),
                           "options": [list(s) for s in q.option_spans], "option_ids": list(q.option_ids)}
                          for q in enc.questions]}


def port_row(row):
    return {"ids": row["ids"], "questions": [{k: q[k] for k in ("qid", "span", "options", "option_ids")}
                                             for q in row["questions"]]}


def main():
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--td", type=int, default=200)
    a = ap.parse_args()
    jsm = ref.upstream(a.snapshot)
    hf = AutoTokenizer.from_pretrained(a.snapshot)
    lay = ClefLayout(tokenizers.Tokenizer.from_file(os.path.join(a.snapshot, "tokenizer.json")), decision_for(jsm))
    same = rejected = shared = 0
    bad = []
    for cid, state, qs in cases.edge_cases() + cases.typed_decisions(a.td):
        if isinstance(qs, dict) and any(ollaya_rule(k, v) for k, v in qs.items()):
            shared += 1
            continue
        try:
            up, ue = upstream_row(ref.encode(hf, state, qs)), None
        except ref.RequestError as e:
            up, ue = None, e
        try:
            ours, oe = port_row(lay.encode(state, qs)), None
        except LayoutError as e:
            ours, oe = None, e
        if ue is not None and oe is not None:
            rejected += 1
        elif (ue is None) != (oe is None):
            bad.append((cid, "rejection differs: upstream %s, port %s" % (ue, oe)))
        elif up != ours:
            u, o = up["ids"], ours["ids"]
            at = next((i for i, (x, y) in enumerate(zip(u, o)) if x != y), min(len(u), len(o)))
            what = "ids: %d vs %d, first difference at %d: %s vs %s" % (
                len(u), len(o), at, u[max(0, at - 3):at + 3], o[max(0, at - 3):at + 3]) if u != o else "spans"
            bad.append((cid, what))
        else:
            same += 1
    print("identical requests %d | both reject %d | Ollaya's shared rules reject %d | mismatches %d"
          % (same, rejected, shared, len(bad)))
    for b in bad[:15]:
        print("  ", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
