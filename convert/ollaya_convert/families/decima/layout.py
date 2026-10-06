"""`decima-late-interaction-v1`: the request handling of the author's `decima/systemone.py` and `decima/model.py`
(github.com/amyrmahdy/decima at v1.1.1), written the way the Rust port is (HF `tokenizers` and `decision.json`
only). `check.py` and `goldens.py` compare it id for id with upstream.

Per question, in request order:

    text     = instructions (a string as is, anything else json.dumps(ensure_ascii=False), the question id when
               they are absent or null); a noul question with a "true" or "false" criterion appends
               "\\nTrue if: <true>\\nFalse if: <false>", an empty side written as an em dash (U+2014)
    options  = choice: "label: description" (the label alone when the description renders empty);
               score: each level's text; noul: "yes", "no"
    state    = the state as is when it is a string, else json.dumps(ensure_ascii=False)
    rows     = [cls] + tok(state_prefix + normalize(text + "\\n" + state))[:max_state_tokens - 2] + [sep]
               [cls] + tok(option_prefix + normalize(text + " " + option))[:max_option_tokens - 2] + [sep], per option

`normalize` is NFC, then Python's str.strip() (upstream's normalize for lang "en", which is all systemone uses);
the empty instructions leave the state alone. A state row over max_state_tokens is cut and flagged (upstream's
server answers 422 STATE_TRUNCATED; its model.py truncates); an option row is cut silently, as upstream does.

Upstream's checks (systemone.py): 1-256 questions, the state a string, object or array, 2-255 choices, 2-10 score
levels, a known type. Option logits come out in upstream's order: a choice's labels, score levels, noul [yes, no].
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict

TYPES = ("choice", "noul", "score")
PLACEHOLDER = re.compile(r"\{(true|false)\}")


class LayoutError(ValueError):
    """The request is invalid for this model (HTTP 422)."""


def text(value: Any) -> str:
    """systemone._text: None as "", a string as is, anything else json.dumps(ensure_ascii=False)."""
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def normalize(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


class DecimaLayout:
    def __init__(self, tokenizer, decision: Dict[str, Any]):
        self.tok = tokenizer
        self.d = decision
        self.cls = decision["special_tokens"]["cls"]
        self.sep = decision["special_tokens"]["sep"]

    def question(self, qid, q):
        """systemone.to_question: (type, text, options)."""
        lim = self.d["limits"]
        if not isinstance(q, dict):
            raise LayoutError("question %r must be an object" % qid)
        t = q.get("type")
        ins = q.get("instructions")
        body = qid if ins is None else text(ins)
        crit = q.get("criteria")
        if t == "choice":
            if isinstance(crit, list):
                crit = {str(c): None for c in dict.fromkeys(crit)}
            if not isinstance(crit, dict) or not (lim["choices"][0] <= len(crit) <= lim["choices"][1]):
                raise LayoutError("question %r: choice needs %d-%d criteria" % (qid, *lim["choices"]))
            return t, body, [("%s: %s" % (k, text(v))) if text(v).strip() else str(k) for k, v in crit.items()]
        if t == "score":
            if not isinstance(crit, list) or not (lim["levels"][0] <= len(crit) <= lim["levels"][1]):
                raise LayoutError("question %r: score needs %d-%d levels" % (qid, *lim["levels"]))
            return t, body, [text(v) for v in crit]
        if t == "noul":
            n = self.d["noul"]
            if isinstance(crit, dict):
                tr, fa = text(crit.get("true")).strip(), text(crit.get("false")).strip()
                if tr or fa:
                    vals = {"true": tr or n["missing"], "false": fa or n["missing"]}
                    body += PLACEHOLDER.sub(lambda m: vals[m.group(1)], n["criteria"])
            return t, body, list(n["options"])
        raise LayoutError("question %r: unknown type %r" % (qid, t))

    def row(self, prefix, s, max_len):
        """-> (ids, cut): the tokenizer's <s> ... </s> around the text, cut to max_len tokens."""
        body = self.tok.encode(prefix + normalize(s), add_special_tokens=False).ids
        return [self.cls] + body[:max_len - 2] + [self.sep], len(body) + 2 > max_len

    def encode(self, state, questions):
        """-> {"state_tokens", "questions": [{"qid", "type", "state_ids", "option_ids", "state_truncated"}]}."""
        lim = self.d["limits"]
        if not isinstance(questions, dict) or not (lim["questions"][0] <= len(questions) <= lim["questions"][1]):
            raise LayoutError("questions: %d-%d questions required" % tuple(lim["questions"]))
        if state is None or isinstance(state, (bool, int, float)):
            raise LayoutError("state: must be a string, object or array")
        s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        out = []
        for qid, q in questions.items():
            t, body, options = self.question(qid, q)
            in_state = body if self.d["question_in_state"] else ""
            in_options = body if self.d["question_in_options"] else ""
            state_ids, cut = self.row(self.d["state_prefix"], "%s\n%s" % (in_state, s) if in_state else s,
                                      self.d["max_state_tokens"])
            option_ids = [self.row(self.d["option_prefix"], ("%s %s" % (in_options, o)).strip(),
                                   self.d["max_option_tokens"])[0] for o in options]
            out.append({"qid": qid, "type": t, "state_ids": state_ids, "option_ids": option_ids,
                        "state_truncated": cut})
        return {"state_tokens": len(self.tok.encode(s, add_special_tokens=False).ids), "questions": out}


def graph_inputs(rows, pad):
    """The graph's inputs for questions' rows (numpy int64): states [q, ls], options [o, lo], option_state [o]."""
    import numpy as np

    ls = max(len(r["state_ids"]) for r in rows)
    lo = max(len(o) for r in rows for o in r["option_ids"])
    n = sum(len(r["option_ids"]) for r in rows)
    state_ids = np.full((len(rows), ls), pad, dtype=np.int64)
    state_mask = np.zeros((len(rows), ls), dtype=np.int64)
    option_ids = np.full((n, lo), pad, dtype=np.int64)
    option_mask = np.zeros((n, lo), dtype=np.int64)
    option_state = np.zeros((n,), dtype=np.int64)
    k = 0
    for i, r in enumerate(rows):
        state_ids[i, :len(r["state_ids"])] = r["state_ids"]
        state_mask[i, :len(r["state_ids"])] = 1
        for o in r["option_ids"]:
            option_ids[k, :len(o)] = o
            option_mask[k, :len(o)] = 1
            option_state[k] = i
            k += 1
    return {"state_ids": state_ids, "state_mask": state_mask, "option_ids": option_ids, "option_mask": option_mask,
            "option_state": option_state}
