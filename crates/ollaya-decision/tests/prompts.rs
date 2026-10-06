//! The llama layouts against their references, case for case.
//!
//! Fixtures: `convert/ollaya_convert/families/llm_common/prompt_goldens.py`, which runs
//! `llm_logits/ref.py`, `winnow/ref.py` and `jevk5/ref.py` on the engine-form requests the daemon
//! sends (the jevk5 prompts also checked against the author's own `jevk5.prompt`), and
//! `convert/ollaya_convert/families/snap/fixture.py`, which takes snap's own prompt export. Token
//! ids and logits need the model; `cargo run -p ollaya-runner --example parity_llama` checks those.

use ollaya_decision::jevk5::{self, JevK5Config};
use ollaya_decision::llm_logits::{self, LlmLogitsConfig};
use ollaya_decision::snap::SnapConfig;
use ollaya_decision::winnow::{self, WinnowConfig};
use ollaya_decision::{Error, QType};
use serde_json::Value;

fn fixture(name: &str) -> String {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn cases(name: &str) -> Vec<Value> {
    fixture(name)
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn qtype(v: &Value) -> QType {
    match v.as_str().unwrap() {
        "choice" => QType::Choice,
        "score" => QType::Score,
        _ => QType::Noul,
    }
}

fn error_class(e: &Error) -> &'static str {
    match e {
        Error::TooManyOptions { .. } => "too_many_options",
        _ => "invalid",
    }
}

#[test]
fn llm_logits_prompts_match_the_reference() {
    let config: LlmLogitsConfig =
        serde_json::from_str(&fixture("llm_logits_decision.json")).unwrap();
    config.validate().unwrap();
    let mut checked = 0;
    for case in cases("llm_logits_prompts.jsonl") {
        let id = case["id"].as_str().unwrap();
        let state_text = llm_logits::render_state(&case["state"]);
        assert_eq!(
            state_text,
            case["state_text"].as_str().unwrap(),
            "{id}: state"
        );
        for (qid, want) in case["expected"].as_object().unwrap() {
            let got = config.question(qid, &case["questions"][qid], &state_text);
            if let Some(err) = want["error"].as_str() {
                let e = got.expect_err(&format!("{id}/{qid}: expected {err}"));
                assert_eq!(error_class(&e), err, "{id}/{qid}: {e}");
                continue;
            }
            let got = got.unwrap_or_else(|e| panic!("{id}/{qid}: {e}"));
            assert_eq!(
                got.user,
                want["user"].as_str().unwrap(),
                "{id}/{qid}: user message"
            );
            assert_eq!(got.qtype, qtype(&want["type"]), "{id}/{qid}: type");
            let ids: Vec<u32> = serde_json::from_value(want["label_ids"].clone()).unwrap();
            assert_eq!(got.label_ids, ids, "{id}/{qid}: label ids");
            let wire: Vec<usize> = serde_json::from_value(want["wire_order"].clone()).unwrap();
            assert_eq!(got.wire_order, wire, "{id}/{qid}: wire order");
            checked += 1;
        }
    }
    assert!(checked > 300, "only {checked} questions checked");
}

#[test]
fn winnow_prompts_match_the_reference() {
    let config: WinnowConfig = serde_json::from_str(&fixture("winnow_decision.json")).unwrap();
    config.validate().unwrap();
    let mut checked = 0;
    for case in cases("winnow_prompts.jsonl") {
        let id = case["id"].as_str().unwrap();
        let got = winnow::state_text(&case["state"])
            .and_then(|s| Ok((s, config.questions(&case["questions"])?)));
        if let Some(err) = case["error"].as_str() {
            let e = got.expect_err(&format!("{id}: expected {err}"));
            assert_eq!(error_class(&e), err, "{id}: {e}");
            continue;
        }
        let (state_text, questions) = got.unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(
            state_text,
            case["state_text"].as_str().unwrap(),
            "{id}: state"
        );
        assert_eq!(
            winnow::prefix(&state_text),
            case["prefix"].as_str().unwrap(),
            "{id}: prefix"
        );
        let want = case["expected"].as_array().unwrap();
        assert_eq!(questions.len(), want.len(), "{id}: questions");
        for ((qid, q), w) in questions.iter().zip(want) {
            assert_eq!(qid, w["qid"].as_str().unwrap(), "{id}: question order");
            assert_eq!(
                q.suffix,
                w["suffix"].as_str().unwrap(),
                "{id}/{qid}: suffix"
            );
            assert_eq!(q.qtype, qtype(&w["type"]), "{id}/{qid}: type");
            let keys: Vec<String> = serde_json::from_value(w["keys"].clone()).unwrap();
            assert_eq!(q.keys, keys, "{id}/{qid}: keys");
            let ids: Vec<u32> = serde_json::from_value(w["label_ids"].clone()).unwrap();
            assert_eq!(q.label_ids, ids, "{id}/{qid}: label ids");
            checked += 1;
        }
    }
    assert!(checked > 200, "only {checked} questions checked");
}

#[test]
fn jevk5_prompts_match_the_reference() {
    let config: JevK5Config = serde_json::from_str(&fixture("jevk5_decision.json")).unwrap();
    config.validate().unwrap();
    let mut checked = 0;
    for case in cases("jevk5_prompts.jsonl") {
        let id = case["id"].as_str().unwrap();
        let got = config.questions(&case["state"], &case["questions"]);
        if let Some(err) = case["error"].as_str() {
            let e = got.expect_err(&format!("{id}: expected {err}"));
            assert_eq!(error_class(&e), err, "{id}: {e}");
            continue;
        }
        let questions = got.unwrap_or_else(|e| panic!("{id}: {e}"));
        let want = case["expected"].as_array().unwrap();
        assert_eq!(questions.len(), want.len(), "{id}: questions");
        for ((qid, q), w) in questions.iter().zip(want) {
            assert_eq!(qid, w["qid"].as_str().unwrap(), "{id}: question order");
            assert_eq!(
                format!("{}{}{}", jevk5::pre(), q.user, jevk5::POST),
                w["prompt"].as_str().unwrap(),
                "{id}/{qid}: prompt"
            );
            assert_eq!(q.qtype, qtype(&w["type"]), "{id}/{qid}: type");
            let keys: Vec<String> = serde_json::from_value(w["keys"].clone()).unwrap();
            assert_eq!(q.keys, keys, "{id}/{qid}: keys");
            let ids: Vec<u32> = serde_json::from_value(w["label_ids"].clone()).unwrap();
            assert_eq!(q.label_ids, ids, "{id}/{qid}: label ids");
            let wire: Vec<usize> = serde_json::from_value(w["wire_order"].clone()).unwrap();
            assert_eq!(q.wire_order, wire, "{id}/{qid}: wire order");
            checked += 1;
        }
    }
    assert!(checked > 400, "only {checked} questions checked");
}

#[test]
fn snap_prompts_match_the_authors_export() {
    let config: SnapConfig = serde_json::from_str(&fixture("snap_decision.json")).unwrap();
    config.validate().unwrap();
    let mut checked = 0;
    for case in cases("snap_prompts.jsonl") {
        let id = case["id"].as_str().unwrap();
        // snap's export asks each question on its own, so each is a request of one question.
        for q in case["questions"].as_array().unwrap() {
            let qid = q["qid"].as_str().unwrap();
            let questions = serde_json::json!({ qid: q["question"].clone() });
            let got = config.questions(&case["state"], &questions);
            if let Some(err) = q["error"].as_str() {
                let e = got.expect_err(&format!("{id}/{qid}: expected {err}"));
                assert_eq!(error_class(&e), err, "{id}/{qid}: {e}");
                continue;
            }
            let (order, mut prompts) = got.unwrap_or_else(|e| panic!("{id}/{qid}: {e}"));
            let want = &q["expected"];
            assert_eq!(order.as_str(), want["layout"], "{id}/{qid}: layout");
            let (_, p) = prompts.remove(0);
            assert_eq!(
                p.user,
                want["user"].as_str().unwrap(),
                "{id}/{qid}: user message"
            );
            let keys: Vec<String> = serde_json::from_value(want["keys"].clone()).unwrap();
            assert_eq!(p.keys, keys, "{id}/{qid}: keys");
            checked += 1;
        }
    }
    assert_eq!(checked, 573, "questions checked");
}
