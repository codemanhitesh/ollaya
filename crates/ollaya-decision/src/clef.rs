//! `clef-joint-v1`: Cloudflare's Clef (a Qwen3.5 backbone with a joint schema head), whose one
//! sequence per request follows upstream `encode_record` in the model repository's
//! `joint_schema_model.py` (`ollaya_convert.families.clef.layout`).
//!
//! ```text
//! ids    = tok(prefix) ⧺ tok(render(state))[..budget] ⧺ schema ⧺ tok(suffix)
//! schema = tok(schema) ⧺ for question i:
//!            tok(field(n, id, type)) ⧺ tok(render(instructions or id))               question span
//!            ⧺ tok(options) ⧺ for option j:
//!                tok(option(n)) ⧺ tok(render({option_id, description})) ⧺ tok(option_end)   option span
//!            ⧺ tok(field_end)
//! ```
//!
//! Every piece is tokenized on its own with special tokens parsed (upstream escapes nothing).
//! `render` keeps a string as is and writes anything else as
//! `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)`; a null description is
//! left out. The options are noul `[true, false]`, a choice's labels sorted by code point, and score
//! levels in order: [`ClefQuestion::in_question_order`] maps the logits back to the order answers
//! use. The state is cut so the sequence fits `max_tokens`.

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::layout::TokenEncoder;
use crate::question::{Criteria, QType, Question};
use crate::{Error, pyjson};

pub const LAYOUT: &str = "clef-joint-v1";

/// The fixed strings of `encode_record`; `field` takes `{n}`, `{id}` and `{type}`, `option` `{n}`.
#[derive(Debug, Clone, Deserialize)]
pub struct Prompt {
    pub prefix: String,
    pub schema: String,
    pub field: String,
    pub options: String,
    pub option: String,
    pub option_end: String,
    pub field_end: String,
    pub suffix: String,
}

/// Upstream's descriptions of the noul options, which a question's criteria replace key by key.
#[derive(Debug, Clone, Deserialize)]
pub struct NoulCriteria {
    pub r#true: Value,
    pub r#false: Value,
}

/// The head's type embedding row of each question type.
#[derive(Debug, Clone, Deserialize)]
pub struct TypeIndex {
    pub noul: i64,
    pub choice: i64,
    pub score: i64,
}

/// The layout as `decision.json` declares it.
#[derive(Debug, Clone, Deserialize)]
pub struct ClefLayout {
    pub prompt: Prompt,
    pub noul_criteria: NoulCriteria,
    pub max_tokens: usize,
    pub type_index: TypeIndex,
    pub pad: u32,
}

/// One question in the sequence: token spans are `[start, end)` positions in [`ClefRow::ids`].
#[derive(Debug, Clone, PartialEq)]
pub struct ClefQuestion {
    pub qid: String,
    pub qtype: QType,
    pub type_index: i64,
    pub span: (usize, usize),
    pub options: Vec<(usize, usize)>,
    pub option_ids: Vec<String>,
}

/// A request's sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct ClefRow {
    pub ids: Vec<u32>,
    pub questions: Vec<ClefQuestion>,
    /// Tokens in the rendered state, before it was cut.
    pub state_tokens: usize,
    pub state_truncated: bool,
}

/// A string as is, anything else as compact, key-sorted JSON.
pub fn render(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => pyjson::dumps_canonical(other),
    }
}

/// Substitute `vars` (`("{n}", "1")`, ...) in one pass over `template`; values are never rescanned.
fn fill(template: &str, vars: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    'scan: while let Some(c) = rest.chars().next() {
        for (name, value) in vars {
            if let Some(r) = rest.strip_prefix(name) {
                out.push_str(value);
                rest = r;
                continue 'scan;
            }
        }
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}

type Options = Vec<(String, Option<Value>)>;

fn described(v: &Value) -> Option<Value> {
    (!v.is_null()).then(|| v.clone())
}

impl ClefLayout {
    pub fn validate(&self) -> Result<(), Error> {
        if self.max_tokens == 0 || self.prompt.suffix.is_empty() {
            return Err(Error::invalid(
                "decision.json: max_tokens and the prompt are required",
            ));
        }
        Ok(())
    }

    /// upstream `question_options` after `systemone`'s checks: the type and `(option id, description)`.
    fn options(&self, qid: &str, def: &Value) -> Result<(QType, i64, Options), Error> {
        let bad = |msg: &str| Error::invalid(format!("question {qid:?}: {msg}"));
        let q = def.as_object().ok_or_else(|| bad("must be an object"))?;
        let crit = q.get("criteria").unwrap_or(&Value::Null);
        match q.get("type").and_then(Value::as_str) {
            Some("noul") => {
                // `criteria.update(question.get("criteria") or {})`
                let (mut t, mut f) = (&self.noul_criteria.r#true, &self.noul_criteria.r#false);
                match crit {
                    Value::Null => {}
                    Value::Object(m) => {
                        t = m.get("true").unwrap_or(t);
                        f = m.get("false").unwrap_or(f);
                    }
                    _ => return Err(bad("noul criteria must be an object")),
                }
                let opts = vec![
                    ("true".into(), described(t)),
                    ("false".into(), described(f)),
                ];
                Ok((QType::Noul, self.type_index.noul, opts))
            }
            Some("choice") => {
                let m = crit
                    .as_object()
                    .filter(|m| !m.is_empty())
                    .ok_or_else(|| {
                        bad("this model takes choice criteria as a non-empty object of label -> description")
                    })?;
                let mut opts: Options = m.iter().map(|(k, v)| (k.clone(), described(v))).collect();
                opts.sort_by(|a, b| a.0.cmp(&b.0));
                Ok((QType::Choice, self.type_index.choice, opts))
            }
            Some("score") => {
                let levels = crit
                    .as_array()
                    .filter(|l| !l.is_empty())
                    .ok_or_else(|| bad("score criteria must be a non-empty list of levels"))?;
                let opts = levels
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i.to_string(), described(v)))
                    .collect();
                Ok((QType::Score, self.type_index.score, opts))
            }
            _ => Err(bad("type must be noul, choice, or score")),
        }
    }

    /// The request's sequence, every question in request order.
    pub fn encode(
        &self,
        tok: &dyn TokenEncoder,
        state: &Value,
        defs: &[(&str, &Value)],
    ) -> Result<ClefRow, Error> {
        if defs.is_empty() {
            return Err(Error::invalid("questions must be a non-empty object"));
        }
        let p = &self.prompt;
        let mut schema = tok.encode(&p.schema)?;
        let mut questions = Vec::with_capacity(defs.len());
        for (i, (qid, def)) in defs.iter().enumerate() {
            let (qtype, type_index, opts) = self.options(qid, def)?;
            let n = (i + 1).to_string();
            let field = fill(
                &p.field,
                &[("{n}", &n), ("{id}", qid), ("{type}", qtype.name())],
            );
            schema.extend(tok.encode(&field)?);
            let start = schema.len();
            let instructions = match def.get("instructions") {
                None | Some(Value::Null) => Value::String((*qid).to_owned()),
                Some(Value::String(s)) if s.is_empty() => Value::String((*qid).to_owned()),
                Some(v) => v.clone(),
            };
            schema.extend(tok.encode(&render(&instructions))?);
            if schema.len() == start {
                return Err(Error::invalid(format!(
                    "question {qid:?}: its instructions are empty"
                )));
            }
            let span = (start, schema.len());
            schema.extend(tok.encode(&p.options)?);
            let mut spans = Vec::with_capacity(opts.len());
            let mut option_ids = Vec::with_capacity(opts.len());
            for (j, (oid, desc)) in opts.into_iter().enumerate() {
                schema.extend(tok.encode(&fill(&p.option, &[("{n}", &(j + 1).to_string())]))?);
                let a = schema.len();
                let mut semantics = Map::new();
                semantics.insert("option_id".into(), Value::String(oid.clone()));
                if let Some(d) = desc {
                    semantics.insert("description".into(), d);
                }
                schema.extend(tok.encode(&render(&Value::Object(semantics)))?);
                spans.push((a, schema.len()));
                option_ids.push(oid);
                schema.extend(tok.encode(&p.option_end)?);
            }
            schema.extend(tok.encode(&p.field_end)?);
            questions.push(ClefQuestion {
                qid: (*qid).to_owned(),
                qtype,
                type_index,
                span,
                options: spans,
                option_ids,
            });
        }
        let prefix = tok.encode(&p.prefix)?;
        let suffix = tok.encode(&p.suffix)?;
        let state_ids = tok.encode(&render(state))?;
        let fixed = prefix.len() + schema.len() + suffix.len();
        if fixed > self.max_tokens {
            return Err(Error::invalid(format!(
                "the questions take {fixed} tokens; this model reads up to {}",
                self.max_tokens
            )));
        }
        let kept = state_ids.len().min(self.max_tokens - fixed);
        let off = prefix.len() + kept;
        for q in &mut questions {
            q.span = (q.span.0 + off, q.span.1 + off);
            for s in &mut q.options {
                *s = (s.0 + off, s.1 + off);
            }
        }
        let mut ids = prefix;
        ids.extend_from_slice(&state_ids[..kept]);
        ids.extend(schema);
        ids.extend(suffix);
        Ok(ClefRow {
            ids,
            questions,
            state_tokens: state_ids.len(),
            state_truncated: kept < state_ids.len(),
        })
    }
}

impl ClefQuestion {
    /// `logits` (one per option, in this layout's order) in the order answers use: a choice's
    /// labels as the caller wrote them, noul `[false, true]`, score levels.
    pub fn in_question_order(&self, q: &Question, logits: &[f32]) -> Result<Vec<f32>, Error> {
        let labels: Vec<&str> = match &q.criteria {
            Criteria::Choice(m) => m.keys().map(String::as_str).collect(),
            Criteria::Noul { .. } => vec!["false", "true"],
            Criteria::Score(levels) => {
                return if levels.len() == logits.len() {
                    Ok(logits.to_vec())
                } else {
                    Err(Error::invalid(format!(
                        "question {:?}: {} logits for {} levels",
                        self.qid,
                        logits.len(),
                        levels.len()
                    )))
                };
            }
        };
        labels
            .iter()
            .map(|l| {
                self.option_ids
                    .iter()
                    .position(|o| o == l)
                    .and_then(|i| logits.get(i).copied())
                    .ok_or_else(|| {
                        Error::invalid(format!(
                            "question {:?}: no logit for option {l:?}",
                            self.qid
                        ))
                    })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::question::parse_questions;
    use serde_json::json;

    /// One token per char, and `<|x|>` (as a special) one token.
    struct Tok;
    impl TokenEncoder for Tok {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
            let mut out = Vec::new();
            let mut rest = text;
            while let Some(c) = rest.chars().next() {
                if let Some(r) = rest.strip_prefix("<|x|>") {
                    out.push(1_000_000);
                    rest = r;
                    continue;
                }
                out.push(c as u32);
                rest = &rest[c.len_utf8()..];
            }
            Ok(out)
        }
    }

    fn layout(max_tokens: usize) -> ClefLayout {
        ClefLayout {
            prompt: Prompt {
                prefix: "<|x|>S\n".into(),
                schema: "\nSCHEMA\n".into(),
                field: "\nFIELD {n}\nID: {id}\nTYPE: {type}\nI: ".into(),
                options: "\nOPTS\n".into(),
                option: "O {n}: ".into(),
                option_end: "\n".into(),
                field_end: "END\n".into(),
                suffix: "\n<|x|>".into(),
            },
            noul_criteria: NoulCriteria {
                r#true: json!("yes"),
                r#false: json!("no"),
            },
            max_tokens,
            type_index: TypeIndex {
                noul: 0,
                choice: 1,
                score: 2,
            },
            pad: 0,
        }
    }

    fn text(row: &ClefRow, span: (usize, usize)) -> String {
        row.ids[span.0..span.1]
            .iter()
            .map(|&t| char::from_u32(t).unwrap_or('?'))
            .collect()
    }

    #[test]
    fn spans_cover_instructions_and_option_semantics() {
        let qs = json!({
            "team": {"type": "choice", "instructions": "Which {id}?", "criteria": {"tech": "bugs", "billing": null}},
            "ok": {"type": "noul", "instructions": "", "criteria": {"false": null}},
            "lvl": {"type": "score", "instructions": {"b": 1, "a": [1.5]}, "criteria": ["low", "high"]},
        });
        let defs: Vec<(&str, &Value)> = qs
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect();
        let row = layout(10_000)
            .encode(&Tok, &json!({"z": 1, "a": "é"}), &defs)
            .unwrap();
        let [team, ok, lvl] = &row.questions[..] else {
            panic!()
        };
        assert_eq!(text(&row, team.span), "Which {id}?");
        assert_eq!(team.option_ids, ["billing", "tech"]);
        assert_eq!(text(&row, team.options[0]), r#"{"option_id":"billing"}"#);
        assert_eq!(
            text(&row, team.options[1]),
            r#"{"description":"bugs","option_id":"tech"}"#
        );
        // empty instructions read the question id; a null description is left out
        assert_eq!(text(&row, ok.span), "ok");
        assert_eq!(
            text(&row, ok.options[0]),
            r#"{"description":"yes","option_id":"true"}"#
        );
        assert_eq!(text(&row, ok.options[1]), r#"{"option_id":"false"}"#);
        assert_eq!(text(&row, lvl.span), r#"{"a":[1.5],"b":1}"#);
        assert_eq!((ok.type_index, team.type_index, lvl.type_index), (0, 1, 2));
        let all: String = row
            .ids
            .iter()
            .map(|&t| char::from_u32(t).unwrap_or('#'))
            .collect();
        assert!(all.contains("FIELD 1\nID: team\nTYPE: choice\nI: Which {id}?"));
        assert!(all.contains("S\n{\"a\":\"é\",\"z\":1}"));
    }

    #[test]
    fn cuts_the_state_to_fit() {
        let qs = json!({"q": {"type": "noul", "instructions": "x"}});
        let defs: Vec<(&str, &Value)> = qs
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect();
        let full = layout(10_000)
            .encode(&Tok, &json!("abcdefghij"), &defs)
            .unwrap();
        let cut = layout(full.ids.len() - 4)
            .encode(&Tok, &json!("abcdefghij"), &defs)
            .unwrap();
        assert!(cut.state_truncated && !full.state_truncated);
        assert_eq!((cut.state_tokens, cut.ids.len()), (10, full.ids.len() - 4));
        assert_eq!(text(&cut, cut.questions[0].span), "x");
        assert!(layout(10).encode(&Tok, &json!("s"), &defs).is_err());
    }

    #[test]
    fn rejects_what_upstream_rejects() {
        let l = layout(10_000);
        for q in [
            json!({"type": "choice", "instructions": "x", "criteria": ["a", "b"]}),
            json!({"type": "choice", "instructions": "x", "criteria": {}}),
            json!({"type": "score", "instructions": "x", "criteria": []}),
            json!({"type": "maybe", "instructions": "x"}),
        ] {
            assert!(l.encode(&Tok, &json!("s"), &[("q", &q)]).is_err(), "{q}");
        }
        // an empty id and no instructions: nothing for the head to read
        let q = json!({"type": "noul", "instructions": null});
        assert!(l.encode(&Tok, &json!("s"), &[("", &q)]).is_err());
    }

    #[test]
    fn logits_come_back_in_question_order() {
        let qs = parse_questions(&json!({
            "c": {"type": "choice", "instructions": "x", "criteria": {"zeta": null, "alpha": null, "mid": null}},
            "n": {"type": "noul", "instructions": "x"},
        }))
        .unwrap();
        let defs: Vec<(&str, &Value)> = qs
            .iter()
            .map(|(k, q)| (k.as_str(), &q.definition))
            .collect();
        let row = layout(10_000).encode(&Tok, &json!("s"), &defs).unwrap();
        // layout order: alpha, mid, zeta; true, false
        let c = row.questions[0]
            .in_question_order(&qs["c"], &[1.0, 2.0, 3.0])
            .unwrap();
        let labels: Vec<&String> = match &qs["c"].criteria {
            Criteria::Choice(m) => m.keys().collect(),
            _ => unreachable!(),
        };
        let want: Vec<f32> = labels
            .iter()
            .map(|l| match l.as_str() {
                "alpha" => 1.0,
                "mid" => 2.0,
                _ => 3.0,
            })
            .collect();
        assert_eq!(c, want);
        assert_eq!(
            row.questions[1]
                .in_question_order(&qs["n"], &[0.9, 0.1])
                .unwrap(),
            [0.1, 0.9]
        );
    }
}
