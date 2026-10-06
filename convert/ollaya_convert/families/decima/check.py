"""The layout port against upstream, id for id, on the shared request set.

    uv run python -m ollaya_convert.families.decima.check [--model decima-small] [--td 400]

Upstream: serve.py's verdict, then decima.systemone.to_question, DecimaConfig.state_of / choice_of and the
checkpoint's tokenizer through transformers, truncated as Decima._encode truncates (`ref.encode`). The port:
`layout.DecimaLayout` on the same tokenizer.json loaded by HF `tokenizers` (what the runtime loads), configured by
`decision_for` below. Every question's state row and option rows must be identical, so must the state's truncation
flag (upstream's 422 STATE_TRUNCATED), and rejections must match. Requests Ollaya's shared question parser rejects
first are counted on their own.

The shared set is `llm_common.cases` (the edge cases and typed-decisions rows); `extra_cases` adds the inputs
Decima's mapping treats specially.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import tokenizers

from ..clef.layout import ollaya_rule
from ..llm_common import cases
from . import ref
from .layout import DecimaLayout, LayoutError

# systemone.to_question's noul rendering, `f"{text}\nTrue if: {tr or M}\nFalse if: {fa or M}"` with M an em dash
# (U+2014), and the options of its `verify` question.
NOUL = {"options": ["yes", "no"], "criteria": "\nTrue if: {true}\nFalse if: {false}", "missing": "\u2014"}


def decision_for(d):
    """The layout's fields of decision.json, from the upstream config and modules."""
    from decima import systemone

    cfg, tok = d.cfg, d.tok
    return {
        "state_prefix": cfg.state_prefix,
        "option_prefix": cfg.choice_prefix,
        "question_in_state": cfg.question_in_state,
        "question_in_options": cfg.question_in_choices,
        "max_state_tokens": cfg.max_state_tokens,
        "max_option_tokens": cfg.max_choice_tokens,
        "special_tokens": {"cls": tok.cls_token_id, "sep": tok.sep_token_id, "pad": tok.pad_token_id},
        "noul": NOUL,
        "limits": {"questions": [1, systemone.MAX_QUESTIONS], "choices": [2, systemone.MAX_CHOICES],
                   "levels": [systemone.MIN_LEVELS, systemone.MAX_LEVELS]},
        "temperature": cfg.temperature,
    }


def wire(x):
    return json.loads(json.dumps(x, ensure_ascii=False))


def extra_cases():
    """Inputs Decima's mapping treats specially, as a JSON client sends them."""
    long_option = "a remarkably long option description " * 12
    qs = {
        "noul_false_only": {"type": "noul", "instructions": "Is the customer angry?", "criteria": {"false": "calm"}},
        "noul_true_object": {"type": "noul", "instructions": "Is money involved?",
                             "criteria": {"true": {"any_of": ["refund", "charge"]}}},
        "noul_null_instructions": {"type": "noul", "instructions": None},
        "choice_numbers": {"type": "choice", "instructions": {"task": "route", "lang": "é"},
                           "criteria": {"a": 1, "b": 2.5, "c": True, "d": None, "e": " ", "f": ["x", "y"]}},
        "choice_long_option": {"type": "choice", "instructions": "Which one?",
                               "criteria": {"long": long_option, "short": "short"}},
        "score_objects": {"type": "score", "instructions": "", "criteria": [{"what": "low"}, "medium", ["high"]]},
        "choice_duplicates": {"type": "choice", "instructions": "Pick", "criteria": ["a", "b", "a", "c"]},
    }
    nfd = "Café über  näive Å ﬁ  "
    return [
        ("decima/specials", wire({"ticket": "Refund my double charge", "n": 3}), wire(qs)),
        ("decima/nfd_state", nfd, wire({"q": {"type": "choice", "instructions": " Résumé? ",
                                              "criteria": {"yes": "ök", "no": None}}})),
        ("decima/list_state", wire([1, "two", {"three": 3.0}]), wire({"q": qs["noul_false_only"]})),
        ("decima/number_state", 42, wire({"q": qs["noul_false_only"]})),
        ("decima/one_choice", "hi", wire({"q": {"type": "choice", "instructions": "x", "criteria": ["only", "only"]}})),
        ("decima/eleven_levels", "hi", wire({"q": {"type": "score", "instructions": "x",
                                                   "criteria": [str(i) for i in range(11)]}})),
        ("decima/256_choices", "hi", wire({"q": {"type": "choice", "instructions": "x",
                                                 "criteria": ["c%d" % i for i in range(256)]}})),
        ("decima/255_choices", "hi", wire({"q": {"type": "choice", "instructions": "x",
                                                 "criteria": ["c%d" % i for i in range(255)]}})),
        ("decima/empty_instructions_state", "", wire({"q": {"type": "noul", "instructions": ""}})),
    ]


def all_cases(td):
    return cases.edge_cases() + cases.typed_decisions(td) + extra_cases()


def shared_rule(questions):
    """Ollaya's shared question parser rejects this request before any layout sees it."""
    return not isinstance(questions, dict) or not questions or any(ollaya_rule(k, v) for k, v in questions.items())


def tokenizer(path):
    tok = tokenizers.Tokenizer.from_file(path)
    tok.no_truncation()
    tok.no_padding()
    return tok


def compare(d, lay, state, questions):
    """-> "identical", "rejected" (both reject) or the first difference."""
    code, msg = ref.verdict(d, state, questions)
    try:
        up, ue = ref.encode(d, state, questions), None
    except ref.RequestError as e:
        up, ue = None, e
    try:
        ours, oe = lay.encode(state, questions), None
    except LayoutError as e:
        ours, oe = None, e
    upstream_rejects = ue is not None or code not in (None, "STATE_TRUNCATED")
    if upstream_rejects or oe is not None:
        if upstream_rejects and oe is not None:
            return "rejected"
        return "rejection differs: upstream %s, port %s" % (ue or code, oe)
    if (code == "STATE_TRUNCATED") != any(q["state_truncated"] for q in ours["questions"]):
        return "STATE_TRUNCATED differs: upstream %s" % code
    for u, o in zip(up, ours["questions"]):
        if (u["qid"], u["type"]) != (o["qid"], o["type"]) or u["state_truncated"] != o["state_truncated"]:
            return "%s: question or truncation differs" % u["qid"]
        if u["state_ids"] != o["state_ids"]:
            a, b = u["state_ids"], o["state_ids"]
            at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            return "%s: state ids %d vs %d, first difference at %d" % (u["qid"], len(a), len(b), at)
        if u["option_ids"] != o["option_ids"]:
            return "%s: option ids differ" % u["qid"]
    return "identical"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=ref.DEFAULT, choices=sorted(ref.MODELS))
    ap.add_argument("--td", type=int, default=400, help="typed-decisions rows (0: all 400)")
    a = ap.parse_args()
    d = ref.load("cpu", a.model)
    lay = DecimaLayout(tokenizer(os.path.join(ref.checkpoint(a.model), "encoder", "tokenizer.json")), decision_for(d))
    counts = {"identical": 0, "rejected": 0, "shared": 0, "truncated": 0}
    bad = []
    for cid, state, questions in all_cases(a.td):
        if shared_rule(questions):
            counts["shared"] += 1
            continue
        r = compare(d, lay, state, questions)
        if r in ("identical", "rejected"):
            counts[r] += 1
            if r == "identical" and ref.verdict(d, state, questions)[0] == "STATE_TRUNCATED":
                counts["truncated"] += 1
        else:
            bad.append((cid, r))
    print("identical requests %(identical)d (%(truncated)d with a truncated state) | both reject %(rejected)d | "
          "Ollaya's shared rules reject %(shared)d" % counts + " | mismatches %d" % len(bad))
    for b in bad[:15]:
        print("  ", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
