//! `snap-v1`: snap1-2b, MiniCPM5-2B with a merged LoRA, read from the option-letter logits after
//! one prompt (`docs/families/snap.md`). A port of snap v0.5.0's prompt compiler
//! (github.com/emnlmn/snap, tag `v0.5.0`, `PROMPT_VERSION` 6):
//!
//! * `src/prompts.rs`: the TOON state rendering, the letter slots, the question block and the user
//!   message;
//! * `src/schema.rs`: what a question may be (`Question::validate`, `DecideRequest::questions`);
//! * `src/engine.rs`: `resolve_layout` and `compile`, for the order of question and state.
//!
//! ```text
//! question_first: QUESTION / instructions / [Answer yes or no. / Yes: / No:] / "" / OPTIONS /
//!                 A) text ... / "" / STATE / state / "" / Reply with one letter only.
//! state_first:    STATE / state / "" / QUESTION ... OPTIONS ... / "" / Reply with one letter only.
//! ids = tok(pre, specials) ++ tok(user, text only) ++ tok(post, specials)
//! ```
//!
//! `pre` and `post` are MiniCPM5's chat template around the user message, with snap's system
//! message and thinking off (`decision.json`). The option logits are the letters' next-token
//! logits; snap pools the bare, space- and newline-prefixed variants of a letter, which this port
//! reads as the bare letter alone. A request is laid out the way snap's `layout: auto` lays it
//! out, as a whole: the state goes first when it is long or when the questions are short next to
//! it. Not ported: snap's extensions to the wire (`boolean` and `numeric` questions, abstain
//! slots, `layout` and `expand`), and choices wider than the 26 letters, which snap expands into
//! one yes/no probe per option. Such a choice is rejected here.
//!
//! The TOON rendering and the slot and message builders are snap's code, adapted. snap is
//! distributed under the MIT license:
//!
//! Copyright (c) 2026 Emanuele Menon
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy of this software
//! and associated documentation files (the "Software"), to deal in the Software without
//! restriction, including without limitation the rights to use, copy, modify, merge, publish,
//! distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
//! Software is furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all copies or
//! substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
//! BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
//! NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//! DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::Error;
use crate::question::QType;

pub const LAYOUT: &str = "snap-v1";
/// `prompts::PROMPT_VERSION` of the snap release this ports.
pub const PROMPT_VERSION: u32 = 6;
/// `prompts::SYSTEM`, verbatim.
pub const SYSTEM: &str = "You are a decision engine. Given a state and a question, you evaluate \
the options and reply with only the letter of the best option. Never explain.";
const LETTERS: &[u8; 26] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
/// `schema::MAX_SLOTS`: one letter per option.
const MAX_SLOTS: usize = 26;
/// `schema::MAX_QUESTIONS`.
const MAX_QUESTIONS: usize = 64;
/// `engine::LONG_STATE_CHARS`: past this rendered length (in bytes) the state goes first.
const LONG_STATE_CHARS: usize = 2000;

#[derive(Debug, Clone, Deserialize)]
pub struct Template {
    /// The template up to the user message, snap's system message included.
    pub pre: String,
    /// The template after the user message, up to the answer.
    pub post: String,
}

/// `decision.json` of a `snap-v1` model (the fields the runtime reads).
#[derive(Debug, Clone, Deserialize)]
pub struct SnapConfig {
    pub layout: String,
    pub template: Template,
    /// `A`..`Z` and their single tokens.
    pub labels: crate::llm_logits::LabelTable,
}

/// Where the question sits relative to the state (`schema::Layout`, the two `auto` picks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    QuestionFirst,
    StateFirst,
}

impl Order {
    pub fn as_str(self) -> &'static str {
        match self {
            Order::QuestionFirst => "question_first",
            Order::StateFirst => "state_first",
        }
    }
}

/// One question as the model reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct QuestionPrompt {
    pub qtype: QType,
    pub user: String,
    pub label_ids: Vec<u32>,
    /// Prompt position of each wire option (noul: `false` is `B. No`, `true` is `A. Yes`).
    pub wire_order: Vec<usize>,
    /// The option keys in prompt order, as snap's `letters` export names them.
    pub keys: Vec<String>,
}

/// A question after `schema::Question`'s parsing and validation.
struct Parsed {
    qtype: QType,
    instructions: String,
    criteria: Option<Value>,
}

/// A slot: the option key and the text shown next to its letter.
struct Slot {
    key: String,
    text: String,
}

impl SnapConfig {
    pub fn validate(&self) -> Result<(), Error> {
        let bad = |msg: String| Err(Error::invalid(format!("decision.json: {msg}")));
        if self.layout != LAYOUT {
            return bad(format!("layout {:?} is not {LAYOUT}", self.layout));
        }
        let letters: Vec<String> = ('A'..='Z').map(String::from).collect();
        if self.labels.strings != letters || self.labels.ids.len() != 26 {
            return bad("labels must be A..Z with one token each".into());
        }
        Ok(())
    }

    /// Every question's prompt, in request order, and the order the request was laid out in. The
    /// request is rejected as a whole when any question is.
    pub fn questions(
        &self,
        state: &Value,
        questions: &Value,
    ) -> Result<(Order, Vec<(String, QuestionPrompt)>), Error> {
        let qs = questions
            .as_object()
            .filter(|q| (1..=MAX_QUESTIONS).contains(&q.len()))
            .ok_or_else(|| {
                Error::invalid(format!(
                    "questions must be an object of 1 to {MAX_QUESTIONS} named questions"
                ))
            })?;
        let mut parsed = Vec::with_capacity(qs.len());
        for (qid, src) in qs {
            if qid.contains('\u{1f}') {
                return Err(Error::invalid(format!(
                    "question name {qid:?}: \\u001f is reserved"
                )));
            }
            parsed.push((qid, parse(qid, src)?));
        }
        let state = render_state_toon(state);
        let order = resolve_layout(parsed.iter().map(|(_, q)| q), state.len());
        let mut out = Vec::with_capacity(parsed.len());
        for (qid, q) in parsed {
            let slots = slots_for(&q);
            let user = user_message(&state, &q, &slots, order);
            if user.contains('\0') {
                return Err(Error::invalid(format!(
                    "question {qid:?}: the prompt would contain a NUL character"
                )));
            }
            let n = slots.len();
            let wire_order = match q.qtype {
                // The wire reads false then true; the prompt shows Yes (A) then No (B).
                QType::Noul => vec![1, 0],
                QType::Choice | QType::Score => (0..n).collect(),
            };
            out.push((
                qid.clone(),
                QuestionPrompt {
                    qtype: q.qtype,
                    user,
                    label_ids: self.labels.ids[..n].to_vec(),
                    wire_order,
                    keys: slots.into_iter().map(|s| s.key).collect(),
                },
            ));
        }
        Ok((order, out))
    }
}

/// `schema::Question` deserialization (`deny_unknown_fields`) and `Question::validate`, for the
/// question types of the wire.
fn parse(qid: &str, src: &Value) -> Result<Parsed, Error> {
    let bad = |msg: &str| Error::invalid(format!("question {qid:?}: {msg}"));
    let src = src.as_object().ok_or_else(|| bad("must be an object"))?;
    if let Some(k) = src
        .keys()
        .find(|k| !matches!(k.as_str(), "type" | "instructions" | "criteria"))
    {
        return Err(bad(&format!(
            "unknown field {k:?}; a question takes type, instructions and criteria"
        )));
    }
    let qtype = match src.get("type").and_then(Value::as_str) {
        Some("choice") => QType::Choice,
        Some("score") => QType::Score,
        Some("noul") => QType::Noul,
        _ => return Err(bad("type must be 'choice', 'score' or 'noul'")),
    };
    // `entry_text`: null means no instructions, other values read as compact JSON.
    let instructions = match src.get("instructions") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    };
    let criteria = src.get("criteria").filter(|c| !c.is_null()).cloned();
    if qtype != QType::Noul {
        let name = if qtype == QType::Choice {
            "choice"
        } else {
            "score"
        };
        let n = match &criteria {
            Some(Value::Object(m)) => m.len(),
            Some(Value::Array(a)) => a.len(),
            _ => return Err(bad(&format!("{name}: criteria required"))),
        };
        if qtype == QType::Score && !matches!(criteria, Some(Value::Array(_))) {
            return Err(bad("score: criteria must be a list of level descriptions"));
        }
        if n < 2 {
            return Err(bad(&format!("{name}: at least 2 options")));
        }
        if qtype == QType::Score && n > MAX_SLOTS {
            return Err(bad(&format!(
                "score: {n} levels exceed {MAX_SLOTS} letter slots"
            )));
        }
        if qtype == QType::Choice && n > MAX_SLOTS {
            // snap expands such a choice into one yes/no probe per option; not ported.
            return Err(Error::TooManyOptions {
                question: qid.to_owned(),
                options: n,
                head_max_len: MAX_SLOTS,
            });
        }
    }
    Ok(Parsed {
        qtype,
        instructions,
        criteria,
    })
}

/// `engine::resolve_layout` for `layout: auto` with no abstain slot: the state goes first when it
/// is long, or when the questions' heads do not outweigh reading the state once per question.
fn resolve_layout<'a>(qs: impl Iterator<Item = &'a Parsed>, state_len: usize) -> Order {
    let (mut n, mut head_sum) = (0usize, 0usize);
    for q in qs {
        n += 1;
        head_sum += q.instructions.len() + q.criteria.as_ref().map_or(0, |c| c.to_string().len());
    }
    let heads_dominate = n <= 1 || head_sum > (n - 1) * state_len;
    if state_len > LONG_STATE_CHARS || !heads_dominate {
        Order::StateFirst
    } else {
        Order::QuestionFirst
    }
}

/// `prompts::option_label`: a choice shows its description, or its key when that is empty.
fn option_label(key: &str, desc: &str) -> String {
    match desc {
        "" => key.to_string(),
        d => d.to_string(),
    }
}

/// `prompts::desc_text`: null is the SDK's "undescribed", which the caller replaces.
fn desc_text(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        other => value_text(other),
    }
}

/// `prompts::value_text`.
fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// `prompts::slots_for`, without abstain and numeric slots.
fn slots_for(q: &Parsed) -> Vec<Slot> {
    let slot = |key: String, text: String| Slot { key, text };
    match q.qtype {
        QType::Noul => vec![
            slot("yes".into(), "Yes".into()),
            slot("no".into(), "No".into()),
        ],
        QType::Choice => match q.criteria.as_ref() {
            Some(Value::Object(m)) => m
                .iter()
                .map(|(k, v)| slot(k.clone(), option_label(k, &desc_text(v))))
                .collect(),
            Some(Value::Array(a)) => a
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let t = desc_text(v);
                    let t = if t.is_empty() {
                        format!("option {}", i + 1)
                    } else {
                        t
                    };
                    slot(t.clone(), t)
                })
                .collect(),
            _ => vec![],
        },
        QType::Score => match q.criteria.as_ref() {
            Some(Value::Array(a)) => a
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let t = desc_text(v);
                    slot(i.to_string(), if t.is_empty() { i.to_string() } else { t })
                })
                .collect(),
            _ => vec![],
        },
    }
}

/// `prompts::question_block`.
fn question_block(q: &Parsed, slots: &[Slot]) -> Vec<String> {
    let mut lines = vec!["QUESTION".to_string()];
    if !q.instructions.is_empty() {
        lines.push(q.instructions.clone());
    }
    if q.qtype == QType::Noul {
        lines.push("Answer yes or no.".into());
        // {"true": ..., "false": ...} criteria describe the two outcomes; the slots keep their
        // names, so P(yes) keeps its meaning.
        if let Some(Value::Object(m)) = &q.criteria {
            for (label, key) in [("Yes", "true"), ("No", "false")] {
                let t = m.get(key).map(desc_text).unwrap_or_default();
                if !t.is_empty() {
                    lines.push(format!("{label}: {t}"));
                }
            }
        }
    }
    lines.push(String::new());
    lines.push("OPTIONS".to_string());
    for (i, slot) in slots.iter().enumerate() {
        lines.push(format!("{}) {}", LETTERS[i] as char, slot.text));
    }
    lines
}

/// `prompts::user_message` for the two layouts `auto` picks.
fn user_message(state: &str, q: &Parsed, slots: &[Slot], order: Order) -> String {
    let qblock = question_block(q, slots);
    let mut lines: Vec<String> = Vec::new();
    match order {
        Order::QuestionFirst => {
            lines.extend(qblock);
            lines.push(String::new());
            lines.push("STATE".to_string());
            lines.push(state.to_string());
        }
        Order::StateFirst => {
            lines.push("STATE".to_string());
            lines.push(state.to_string());
            lines.push(String::new());
            lines.extend(qblock);
        }
    }
    lines.push(String::new());
    lines.push("Reply with one letter only.".to_string());
    lines.join("\n")
}

// ------------------------------------------------------------------------------------------------
// TOON (spec v4.1, as snap pins it): `prompts::render_state_toon` and its helpers. The same JSON
// data model with declared array lengths `[N]` and per-table field lists `{f1,f2}` instead of
// repeated keys. Encode only.

/// Shortest round-trip formatting of a float, as snap writes numbers.
fn fmt_g(v: f64) -> String {
    format!("{v}")
}

/// A state as snap renders it: a string as it is, anything else in TOON.
pub fn render_state_toon(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => {
            let mut s = String::new();
            toon_root(other, &mut s);
            s.trim_end().to_string()
        }
    }
}

/// A header field-list entry: a bare name, or a name carrying a nested group (`temp{min,max}`)
/// for a column of nested-uniform objects.
enum Field {
    Leaf(String),
    Group(String, Vec<Field>),
}

/// The column walk (spec 9.3 and 9.5): all objects non-empty with the same key set, every column
/// uniform-primitive or nested-uniform. The field list follows the first object's key order;
/// `None` when the objects are not tabular.
fn fields_of(objs: &[&Map<String, Value>]) -> Option<Vec<Field>> {
    let first = objs.first()?;
    if first.is_empty() {
        return None;
    }
    let keys: Vec<&String> = first.keys().collect();
    let uniform = objs
        .iter()
        .all(|m| m.len() == keys.len() && keys.iter().all(|k| m.contains_key(*k)));
    if !uniform {
        return None;
    }
    keys.iter()
        .map(|k| {
            let col: Vec<&Value> = objs.iter().map(|m| &m[k.as_str()]).collect();
            if col
                .iter()
                .all(|v| v.as_object().is_some_and(|o| !o.is_empty()))
            {
                let subs: Vec<&Map<String, Value>> =
                    col.iter().filter_map(|v| v.as_object()).collect();
                Some(Field::Group(k.to_string(), fields_of(&subs)?))
            } else if col.iter().all(|v| !v.is_object() && !v.is_array()) {
                Some(Field::Leaf(k.to_string()))
            } else {
                None
            }
        })
        .collect()
}

/// Tabular detection on an array of objects (spec 9.3).
fn tabular_fields(a: &[Value]) -> Option<Vec<Field>> {
    if a.is_empty() {
        return None;
    }
    let objs: Vec<&Map<String, Value>> = a.iter().map(|v| v.as_object()).collect::<Option<_>>()?;
    fields_of(&objs)
}

/// Keyed tabular detection on an object (spec 9.5): two entries or more, every value a non-empty
/// object, uniform columns across them.
fn keyed_fields(m: &Map<String, Value>) -> Option<Vec<Field>> {
    if m.len() < 2 {
        return None;
    }
    let objs: Vec<&Map<String, Value>> =
        m.values().map(|v| v.as_object()).collect::<Option<_>>()?;
    fields_of(&objs)
}

fn field_list(fields: &[Field]) -> String {
    fields
        .iter()
        .map(|f| match f {
            Field::Leaf(k) => toon_key(k),
            Field::Group(k, subs) => format!("{}{{{}}}", toon_key(k), field_list(subs)),
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Row cells in depth-first leaf order over the field list.
fn cells_of<'a>(m: &'a Map<String, Value>, fields: &[Field], out: &mut Vec<&'a Value>) {
    for f in fields {
        match f {
            Field::Leaf(k) => out.push(&m[k.as_str()]),
            Field::Group(k, subs) => {
                if let Some(sub) = m[k.as_str()].as_object() {
                    cells_of(sub, subs, out);
                }
            }
        }
    }
}

fn row_cells(m: &Map<String, Value>, fields: &[Field]) -> String {
    let mut cells = Vec::new();
    cells_of(m, fields, &mut cells);
    cells
        .iter()
        .map(|v| toon_scalar(v))
        .collect::<Vec<_>>()
        .join(",")
}

/// Keys are bare only for `^[A-Za-z_][A-Za-z0-9_.]*$` (spec 7.3).
fn toon_key(k: &str) -> String {
    let bare = k.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && k.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.');
    if bare { k.to_string() } else { toon_quote(k) }
}

/// The five escapes plus `\uXXXX` for the other controls (spec 7.1).
fn toon_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Strings that look like numbers (`^[+-]?[0-9]+(\.[0-9]+)?(e[+-]?[0-9]+)?$`, spec 4) are quoted.
fn numeric_like(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = (b.first() == Some(&b'+') || b.first() == Some(&b'-')) as usize;
    let d0 = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == d0 {
        return false;
    }
    if b.get(i) == Some(&b'.') {
        i += 1;
        let d1 = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == d1 {
            return false;
        }
    }
    if matches!(b.get(i), Some(&b'e') | Some(&b'E')) {
        i += 1;
        if matches!(b.get(i), Some(&b'+') | Some(&b'-')) {
            i += 1;
        }
        let d2 = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == d2 {
            return false;
        }
    }
    i == b.len()
}

/// Bare only when quoting is not required (spec 7.2). The comma is both the document and the
/// active delimiter everywhere, so it always forces quotes.
fn toon_bare(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with([' ', '\t', '-', '#'])
        && !s.ends_with([' ', '\t'])
        && !s.contains(|c: char| {
            matches!(c, ':' | '"' | '\\' | '[' | ']' | '{' | '}' | ',') || (c as u32) < 0x20
        })
        && !matches!(s, "true" | "false" | "null")
        && !numeric_like(s)
}

fn toon_scalar(v: &Value) -> String {
    match v {
        Value::String(s) if toon_bare(s) => s.clone(),
        Value::String(s) => toon_quote(s),
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => i.to_string(),
            (None, Some(u)) => u.to_string(),
            (None, None) => {
                let v = n.as_f64().unwrap_or_default();
                if v == 0.0 {
                    // -0 is written as 0 (spec 2).
                    "0".into()
                } else {
                    fmt_g(v)
                }
            }
        },
        other => toon_quote(&value_text(other)),
    }
}

/// One object field. `pre` is the line prefix (the indent, or the indent and `- ` for a list
/// item's first field, spec 10); `ind` is the field's own depth.
fn toon_field(pre: &str, k: &str, val: &Value, ind: usize, out: &mut String) {
    let key = toon_key(k);
    match val {
        Value::Object(m) if m.is_empty() => out.push_str(&format!("{pre}{key}:\n")),
        Value::Object(m) => match keyed_fields(m) {
            Some(fields) => {
                out.push_str(&format!(
                    "{pre}{key}[{}:]{{{}}}:\n",
                    m.len(),
                    field_list(&fields)
                ));
                for (ek, ev) in m {
                    let row = ev
                        .as_object()
                        .map(|o| row_cells(o, &fields))
                        .unwrap_or_default();
                    out.push_str(&format!(
                        "{}{}: {}\n",
                        "  ".repeat(ind + 1),
                        toon_key(ek),
                        row
                    ));
                }
            }
            None => {
                out.push_str(&format!("{pre}{key}:\n"));
                for (k2, v2) in m {
                    toon_field(&"  ".repeat(ind + 1), k2, v2, ind + 1, out);
                }
            }
        },
        Value::Array(a) => toon_array(pre, &key, a, ind, out),
        _ => out.push_str(&format!("{pre}{key}: {}\n", toon_scalar(val))),
    }
}

/// `key` is empty only at the root: a keyless header `[N]:` or `[N]{f}:`.
fn toon_array(pre: &str, key: &str, a: &[Value], ind: usize, out: &mut String) {
    if a.is_empty() {
        // `key: []` in field position, `[]` at the root (spec 9.1).
        if key.is_empty() {
            out.push_str("[]\n");
        } else {
            out.push_str(&format!("{pre}{key}: []\n"));
        }
        return;
    }
    if a.iter().all(|v| !v.is_object() && !v.is_array()) {
        let vals: Vec<String> = a.iter().map(toon_scalar).collect();
        out.push_str(&format!("{pre}{key}[{}]: {}\n", a.len(), vals.join(",")));
        return;
    }
    if let Some(fields) = tabular_fields(a) {
        out.push_str(&format!(
            "{pre}{key}[{}]{{{}}}:\n",
            a.len(),
            field_list(&fields)
        ));
        for x in a {
            let row = x
                .as_object()
                .map(|o| row_cells(o, &fields))
                .unwrap_or_default();
            out.push_str(&format!("{}{}\n", "  ".repeat(ind + 1), row));
        }
        return;
    }
    out.push_str(&format!("{pre}{key}[{}]:\n", a.len()));
    for x in a {
        toon_item(x, ind + 1, out);
    }
}

/// List items at depth `ind` (spec 9.4); an object item carries its first field on the hyphen
/// line and its scope content at `ind + 2` (spec 10).
fn toon_item(v: &Value, ind: usize, out: &mut String) {
    let pad = "  ".repeat(ind);
    match v {
        Value::Object(m) if m.is_empty() => out.push_str(&format!("{pad}-\n")),
        Value::Object(m) => {
            let mut it = m.iter();
            if let Some((k, val)) = it.next() {
                toon_field(&format!("{pad}- "), k, val, ind + 1, out);
            }
            for (k, val) in it {
                toon_field(&format!("{pad}  "), k, val, ind + 1, out);
            }
        }
        Value::Array(inner) if inner.is_empty() => out.push_str(&format!("{pad}- [0]:\n")),
        Value::Array(inner) if inner.iter().all(|x| !x.is_object() && !x.is_array()) => {
            let vals: Vec<String> = inner.iter().map(toon_scalar).collect();
            out.push_str(&format!("{pad}- [{}]: {}\n", inner.len(), vals.join(",")));
        }
        Value::Array(inner) => {
            // Keyless headers with fields are not valid at an item position (spec 6), so nested
            // uniform arrays use the list form, never the tabular one.
            out.push_str(&format!("{pad}- [{}]:\n", inner.len()));
            for x in inner {
                toon_item(x, ind + 1, out);
            }
        }
        _ => out.push_str(&format!("{pad}- {}\n", toon_scalar(v))),
    }
}

fn toon_root(v: &Value, out: &mut String) {
    match v {
        // An empty object is an empty document.
        Value::Object(m) if m.is_empty() => {}
        Value::Object(m) => match keyed_fields(m) {
            Some(fields) => {
                out.push_str(&format!("[{}:]{{{}}}:\n", m.len(), field_list(&fields)));
                for (ek, ev) in m {
                    let row = ev
                        .as_object()
                        .map(|o| row_cells(o, &fields))
                        .unwrap_or_default();
                    out.push_str(&format!("  {}: {}\n", toon_key(ek), row));
                }
            }
            None => {
                for (k, v2) in m {
                    toon_field("", k, v2, 0, out);
                }
            }
        },
        Value::Array(a) => toon_array("", "", a, 0, out),
        _ => out.push_str(&format!("{}\n", toon_scalar(v))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config() -> SnapConfig {
        SnapConfig {
            layout: LAYOUT.into(),
            template: Template {
                pre: "<s><|im_start|>system\n".into(),
                post: "<|im_end|>\n<|im_start|>assistant\n".into(),
            },
            labels: crate::llm_logits::LabelTable {
                strings: ('A'..='Z').map(String::from).collect(),
                ids: (200..226).collect(),
            },
        }
    }

    fn one(state: Value, q: Value) -> Result<(Order, QuestionPrompt), Error> {
        let (order, mut qs) = config().questions(&state, &json!({ "q": q }))?;
        Ok((order, qs.remove(0).1))
    }

    #[test]
    fn question_first_for_a_short_state() {
        let (order, p) = one(
            json!("hello world"),
            json!({"type": "noul", "instructions": "Refund?",
                   "criteria": {"true": "explicit money-back request", "false": "anything else"}}),
        )
        .unwrap();
        assert_eq!(order, Order::QuestionFirst);
        assert_eq!(
            p.user,
            "QUESTION\nRefund?\nAnswer yes or no.\nYes: explicit money-back request\nNo: anything else\n\n\
             OPTIONS\nA) Yes\nB) No\n\nSTATE\nhello world\n\nReply with one letter only."
        );
        assert_eq!(p.wire_order, [1, 0]);
        assert_eq!(p.keys, ["yes", "no"]);
        assert_eq!(p.label_ids, [200, 201]);
    }

    #[test]
    fn slots_follow_snap() {
        // An empty or null description shows the key; a list's null entry gets "option i".
        let (_, p) = one(
            json!("s"),
            json!({"type": "choice", "criteria": {"a": null, "b": "bee", "c": ""}}),
        )
        .unwrap();
        assert!(
            p.user.contains("OPTIONS\nA) a\nB) bee\nC) c\n"),
            "{}",
            p.user
        );
        assert_eq!(p.keys, ["a", "b", "c"]);
        let (_, p) = one(
            json!("s"),
            json!({"type": "choice", "criteria": ["x", null]}),
        )
        .unwrap();
        assert_eq!(p.keys, ["x", "option 2"]);
        // A score level shows its index when undescribed, and JSON instructions read compact.
        let (_, p) = one(
            json!("s"),
            json!({"type": "score", "instructions": {"q": 1}, "criteria": ["low", null, "high"]}),
        )
        .unwrap();
        assert!(
            p.user
                .starts_with("QUESTION\n{\"q\":1}\n\nOPTIONS\nA) low\nB) 1\nC) high\n"),
            "{}",
            p.user
        );
        assert_eq!(p.keys, ["0", "1", "2"]);
    }

    #[test]
    fn layout_is_decided_for_the_whole_request() {
        let c = config();
        let long = "x".repeat(LONG_STATE_CHARS + 1);
        let (order, _) = c
            .questions(&json!(long), &json!({"q": {"type": "noul"}}))
            .unwrap();
        assert_eq!(order, Order::StateFirst);
        // Two short questions on a state longer than their heads: the state goes first.
        let (order, qs) = c
            .questions(
                &json!("a state that is longer than both questions together"),
                &json!({"a": {"type": "noul", "instructions": "Yes?"},
                        "b": {"type": "noul", "instructions": "No?"}}),
            )
            .unwrap();
        assert_eq!(order, Order::StateFirst);
        assert!(qs[1].1.user.starts_with("STATE\na state"));
    }

    #[test]
    fn rejects_what_snap_rejects() {
        for q in [
            json!({"type": "choice", "criteria": {"only": "one"}}),
            json!({"type": "choice"}),
            json!({"type": "score", "criteria": {"a": 1, "b": 2}}),
            json!({"type": "boolean"}),
            json!({"type": "noul", "allow_abstain": true}),
            json!({"type": "noul", "instructions": "a\u{0}b"}),
        ] {
            assert!(
                matches!(one(json!("s"), q.clone()), Err(Error::Invalid(_))),
                "{q}"
            );
        }
        let wide: Vec<String> = (0..27).map(|i| format!("o{i}")).collect();
        assert!(matches!(
            one(json!("s"), json!({"type": "choice", "criteria": wide})),
            Err(Error::TooManyOptions { options: 27, .. })
        ));
        assert!(matches!(
            config().questions(&json!("s"), &json!({"a\u{1f}b": {"type": "noul"}})),
            Err(Error::Invalid(_))
        ));
    }

    #[test]
    fn toon_forms() {
        let v = json!({"user": {"name": "ada", "n": 3}, "tags": ["x", "y"], "note": "see: this"});
        let s = render_state_toon(&v);
        assert!(s.contains("user:\n  name: ada\n  n: 3"), "{s}");
        assert!(s.contains("tags[2]: x,y"), "{s}");
        assert!(s.contains("note: \"see: this\""), "{s}");
        let v = json!({"items": [{"sku": "a", "qty": 1}, {"sku": "b", "qty": 2}]});
        assert_eq!(render_state_toon(&v), "items[2]{sku,qty}:\n  a,1\n  b,2");
        let v = json!({"fc": [{"d": "Mon", "t": {"min": -2, "max": 4}}]});
        assert_eq!(render_state_toon(&v), "fc[1]{d,t{min,max}}:\n  Mon,-2,4");
        let v = json!({"env": {"prod": {"r": 1, "d": false}, "stg": {"r": 2, "d": true}}});
        assert_eq!(
            render_state_toon(&v),
            "env[2:]{r,d}:\n  prod: 1,false\n  stg: 2,true"
        );
        let v = json!({"xs": [1, {"a": 2, "b": 3}]});
        assert_eq!(render_state_toon(&v), "xs[2]:\n  - 1\n  - a: 2\n    b: 3");
        let v = json!({"xs": [{"rows": [{"a": 1}], "k": 2}]});
        assert_eq!(
            render_state_toon(&v),
            "xs[1]:\n  - rows[1]{a}:\n      1\n    k: 2"
        );
        let v = json!({"o": {}, "a": [], "n": null});
        assert_eq!(render_state_toon(&v), "o:\na: []\nn: null");
        assert_eq!(render_state_toon(&json!([1, 2])), "[2]: 1,2");
        assert_eq!(
            render_state_toon(&json!([{"a": 1}, {"a": 2}])),
            "[2]{a}:\n  1\n  2"
        );
    }

    #[test]
    fn toon_quoting() {
        let cases = [
            ("", "\"\""),
            (" x", "\" x\""),
            ("x ", "\"x \""),
            ("true", "\"true\""),
            ("42", "\"42\""),
            ("+5", "\"+5\""),
            ("05", "\"05\""),
            ("1e-6", "\"1e-6\""),
            ("a,b", "\"a,b\""),
            ("a:b", "\"a:b\""),
            ("a[b", "\"a[b\""),
            ("-x", "\"-x\""),
            ("#c", "\"#c\""),
            ("a\tb", "\"a\\tb\""),
            ("a\nb", "\"a\\nb\""),
            ("hi there", "hi there"),
        ];
        for (s, want) in cases {
            let got = render_state_toon(&json!({"k": s}));
            assert_eq!(got, format!("k: {want}"), "input {s:?}");
        }
        assert_eq!(render_state_toon(&json!({"my key": 1})), "\"my key\": 1");
        assert_eq!(render_state_toon(&json!({"a.b_c": 1})), "a.b_c: 1");
        assert_eq!(render_state_toon(&json!({"v": -0.0})), "v: 0");
        assert_eq!(render_state_toon(&json!("plain")), "plain");
    }
}
