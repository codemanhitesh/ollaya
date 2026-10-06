"""The `snap-v1` prompt fixture, from snap's own prompt export.

    python -m ollaya_convert.families.snap.fixture <requests.jsonl> <snap-export.jsonl> <out-dir>

`requests.jsonl` is the shared request set, one `{"id", "state", "questions"}` per line.
`snap-export.jsonl` is `snap export-prompts` (snap v0.5.0, `PROMPT_VERSION` 6) on that set, split
into one case per question (`<request id>#<question>`), as the author posted it on issue #46. The
port is checked against the author's code, so there is no Python reference: snap is Rust.

Writes `snap_prompts.jsonl` (per request: its state, and per question, asked on its own as snap's
export asks it, the user message, the layout and the option keys snap rendered, or the error class
of a question snap rejects) and
`snap_decision.json` (the template around the user message) for
`crates/ollaya-decision/tests/prompts.rs`. A choice wider than 26 letters, which snap expands into
yes/no probes, is expected to be rejected (`too_many_options`): the expansion is not ported.
Standard library only.
"""
from __future__ import annotations

import json
import os
import sys

USER = "<|im_start|>user\n"
END = "<|im_end|>\n<|im_start|>assistant"


def split(prompt):
    i = prompt.index(USER) + len(USER)
    j = prompt.rindex(END)
    return prompt[:i], prompt[i:j], prompt[j:]


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    requests = {r["id"]: r for r in map(json.loads, open(sys.argv[1], encoding="utf-8"))}
    frames, cases, seen = set(), {}, set()
    for line in open(sys.argv[2], encoding="utf-8"):
        r = json.loads(line)
        rid, _, qid = r["id"].partition("#")
        qid = qid.split("#")[0]
        if (rid, qid) in seen:
            continue  # the further probes of one expanded choice
        seen.add((rid, qid))
        req = requests[rid]
        if rid not in cases:
            cases[rid] = {"id": rid, "state": req["state"], "questions": []}
        case = {"qid": qid, "question": req["questions"][qid]}
        if "expand" in r:
            case["error"] = "too_many_options"
        elif "error" in r:
            case["error"] = "invalid"
            case["snap_error"] = r["error"]
        else:
            pre, user, post = split(r["prompt"])
            frames.add((pre, post))
            case["expected"] = {"user": user, "layout": r["layout"],
                                "keys": [s["key"] for s in r["letters"]]}
        cases[rid]["questions"].append(case)
    if len(frames) != 1:
        raise SystemExit("expected one template frame, found %d" % len(frames))
    (pre, post), = frames
    out = sys.argv[3]
    with open(os.path.join(out, "snap_prompts.jsonl"), "w", encoding="utf-8") as f:
        for c in cases.values():
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    decision = {"layout": "snap-v1", "template": {"pre": pre, "post": post},
                "labels": {"strings": [chr(65 + i) for i in range(26)], "ids": list(range(26))}}
    with open(os.path.join(out, "snap_decision.json"), "w", encoding="utf-8") as f:
        json.dump(decision, f, ensure_ascii=False, indent=1)
        f.write("\n")
    qs = [q for c in cases.values() for q in c["questions"]]
    n = {k: sum(1 for q in qs if (q.get("error") or "ok") == k) for k in ("ok", "invalid", "too_many_options")}
    print("%d requests, %d questions: %s" % (len(cases), len(qs), n))


if __name__ == "__main__":
    main()
