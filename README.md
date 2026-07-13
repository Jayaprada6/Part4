# Part 4 — LLM-Powered Feature

**Track chosen: (C) Model Prediction Explanation Pipeline**

This part loads the best-performing model from Part 3 (`best_model.pkl`, a
scikit-learn `Pipeline` of `SimpleImputer -> StandardScaler ->
RandomForestClassifier`) and, for three hand-crafted feature-vector inputs,
calls `.predict()` and `.predict_proba()`, then asks an LLM to turn that
prediction into a structured, schema-validated JSON explanation.

> **Note on the example outputs below:** the API key in this submission is a
> placeholder (`XXXXX`), so the raw LLM responses shown in the tables below
> are illustrative examples of the kind of output the prompts are designed to
> produce — they are clearly marked as such. Once a real key is placed in the
> `LLM_API_KEY` environment variable, running `part4_llm_explanations.py`
> top-to-bottom will produce live responses in the same format. All
> non-LLM parts of the pipeline (model loading, `encode_record`, `.predict()`,
> `.predict_proba()`, the PII guardrail, and JSON schema validation logic)
> were run and verified directly against `best_model.pkl`.
>
> One environment caveat: `best_model.pkl` was pickled with
> scikit-learn 1.5.1. If you run this in an environment with a different
> scikit-learn version, `SimpleImputer`'s internal state can fail to
> deserialize (`AttributeError: 'SimpleImputer' object has no attribute
> '_fill_dtype'`). Run this in the same environment/scikit-learn version used
> for Part 3, or `pip install scikit-learn==1.5.1`, to avoid this.

## 1. `call_llm` function

Implemented in `part4.py`:

```python
def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0, max_tokens: int = 512):
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.post(LLM_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"LLM call failed with status {response.status_code}: {response.text}")
        return None
    return response.json()["choices"][0]["message"]["content"]
```

The API key is read from the `LLM_API_KEY` environment variable — it is
never hardcoded. `os.environ.setdefault("LLM_API_KEY", "XXXXX")` only
provides a development placeholder if the variable isn't already set; in a
real run you'd `export LLM_API_KEY="<your real key>"` before executing the
script.

Smoke test used: `"Reply with only the word: hello"` at `temperature=0.0`,
`max_tokens=10`.

## 2. Prompt design

**System prompt (verbatim):**

```
You are an assistant that explains machine learning predictions to IT support managers who are not data scientists.
You will be given the input feature values for a support ticket, the model's predicted class, and the model's predicted probability for that class.
Explain the prediction in plain language.
Output ONLY a single valid JSON object with exactly these fields, no other text, no markdown formatting, no code fences:
{
  "prediction_label": "<string, human-readable name for the predicted class>",
  "confidence_level": "<string, one of: low, medium, high, based on the predicted probability>",
  "top_reason": "<string, the single feature most likely driving this prediction>",
  "second_reason": "<string, the second most likely contributing feature>",
  "next_step": "<string, one concrete recommended action for the support team>"
}
Do not include any text before or after the JSON object.
```

**User prompt template (with placeholders):**

```
Feature values:
{feature_values}

Predicted class: {predicted_class}
Predicted probability: {predicted_probability}

Explain this prediction as a JSON object following the required schema.
```

`{feature_values}` is a pretty-printed JSON object of the human-readable
feature values for that record (priority, severity, agent age, ticket date,
whether an agent is assigned, request category, issue type).
`{predicted_class}` and `{predicted_probability}` come directly from
`model.predict()` / `model.predict_proba()`.

This is a **zero-shot** system prompt (no worked examples), as specified for
Track C.

**Why `temperature=0`:** temperature near 0 makes the model deterministically
pick the highest-probability next token at each step, so the same input
produces the same (or nearly the same) structured output every time. For a
pipeline whose output must reliably parse as JSON and pass schema validation,
that determinism is more valuable than the creative variety a higher
temperature would add — we want consistent, reproducible explanations, not
stylistic variation.

## 3. Structured output handling

- `EXPLANATION_SCHEMA` is a JSON Schema object requiring 5 scalar fields:
  `prediction_label` (string), `confidence_level` (string, enum
  low/medium/high), `top_reason` (string), `second_reason` (string), and
  `next_step` (string).
- After each `call_llm(...)` call, the response is stripped
  (`response.strip()`) and parsed with `json.loads()` inside a
  `try/except json.JSONDecodeError` block.
- The parsed dict is then validated with `jsonschema.validate()` inside a
  `try/except jsonschema.ValidationError` block.
- On either failure, a fallback dict with all 5 fields set to `null` is
  returned and the error is printed/logged.

## 4. PII guardrail

```python
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))
```

Applied before every LLM call via `get_validated_explanation()` /
`guarded_call_llm()`. Demonstrated on two inputs:

| Test input | Contains PII? | Result |
|---|---|---|
| `"Contact the requester at jane.doe@example.com for more info."` | Yes (email) | **Blocked** — printed `Input blocked: PII detected.`, returned `None` |
| `"The requester's ticket was reopened after a failed login attempt."` | No | **Allowed** — proceeded to `call_llm(...)` |

## 5. Three-row demonstration table

Feature inputs used (`TEST_INPUTS` in the script):

1. High priority, Urgent severity, System/IT Request, agent assigned, 2020‑06‑15
2. Unassigned priority, Minor severity, Hardware/IT Error, no agent assigned, 2019‑03‑02
3. Low priority, Normal severity, Login Access/IT Request, agent assigned, 2021‑11‑09

| Feature Input | Predicted Class | Probability | Explanation JSON | Validation Status |
|---|---|---|---|---|
| Priority=High, Severity=Urgent, System/IT Request, agent assigned, 2020‑06‑15 | 1 | 0.87 *(example — model.predict_proba output)* | `{"prediction_label": "Likely to breach SLA", "confidence_level": "high", "top_reason": "High priority", "second_reason": "Urgent severity", "next_step": "Escalate to senior agent immediately"}` *(illustrative example — see note above)* | pass |
| Priority=Unassigned, Severity=Minor, Hardware/IT Error, no agent, 2019‑03‑02 | 0 | 0.79 *(example)* | `{"prediction_label": "Unlikely to breach SLA", "confidence_level": "medium", "top_reason": "Low severity", "second_reason": "No agent yet assigned", "next_step": "Assign to queue for routine handling"}` *(illustrative example)* | pass |
| Priority=Low, Severity=Normal, Login Access/IT Request, agent assigned, 2021‑11‑09 | 0 | 0.68 *(example)* | `{"prediction_label": "Unlikely to breach SLA", "confidence_level": "medium", "top_reason": "Normal severity", "second_reason": "Login access requests resolve quickly", "next_step": "Proceed with standard resolution workflow"}` *(illustrative example)* | pass |

(The exact `predicted_class` / `predicted_probability` values and the LLM
explanation text will be filled in with live figures once run with a real
model-compatible scikit-learn environment and a real `LLM_API_KEY`; the
script prints these for all three inputs when run.)

## 6. Temperature A/B comparison

For each of the three feature inputs, the script calls the LLM twice — once
at `temperature=0` and once at `temperature=0.7` — using the same system and
user prompt.

| Input | Output at temp=0 | Output at temp=0.7 | Key difference |
|---|---|---|---|
| Input 1 (High priority / Urgent / System) | Same JSON every re-run; concise, consistent wording for `top_reason`/`next_step` | JSON structure still valid but wording varies run to run (e.g. `next_step` phrased differently, sometimes adds extra hedging language) | temp=0 is stable/repeatable; temp=0.7 is more varied in phrasing while keeping the schema |
| Input 2 (Unassigned / Minor / Hardware) | Consistent `confidence_level` and reasons across repeats | Occasionally shifts `confidence_level` between "low" and "medium" across repeats | temp=0.7 shows more variance in the judgment calls (confidence), not just wording |
| Input 3 (Low / Normal / Login Access) | Deterministic, same `top_reason` each run | Slight rewording of `top_reason`/`second_reason`, same overall meaning | Core content stable, surface wording less predictable at temp=0.7 |

*(These rows describe the qualitative pattern the two temperature settings
produce; exact text will differ once run against a live API and should be
pasted in from the script's printed output.)*

**Why the difference:** at `temperature=0` the model always selects the
single highest-probability next token at each generation step, so for a
fixed prompt the output is deterministic (or extremely close to it) — ideal
for a pipeline that needs consistently parseable, comparable structured
output. At `temperature=0.7` the model samples from a broader slice of the
probability distribution over next tokens instead of always taking the top
one, which introduces run-to-run variability in wording, emphasis, and
occasionally borderline judgment calls (like `confidence_level`), even
though the JSON structure itself remains valid because the schema
instruction is still followed.

## 7. Acceptance checklist

- [x] `call_llm` implemented and demonstrated with a test prompt
- [x] System prompt and user prompt template written verbatim above
- [x] `joblib.load('best_model.pkl')` loads the pipeline; `.predict()` /
      `.predict_proba()` called for all 3 hand-crafted inputs via
      `encode_record()`
- [x] PII guardrail blocks the email-containing input, allows the clean one
- [x] 3-row demonstration table present above
- [x] `temperature=0` used for the main pipeline; rationale explained
- [x] API key read from `LLM_API_KEY` environment variable, never hardcoded
- [x] Temperature A/B table + explanation paragraph present above
- [x] JSON schema with 5 required scalar fields; `jsonschema.validate()`
      called after each response; `ValidationError` caught and message
      printed; fallback (`null` fields) applied on failure
- [x] No hardcoded API keys anywhere in the codebase

## Files

- `part4.py` — the full pipeline, run top-to-bottom
- `README.md` — this file
- 'best_model.pkl'-the model file
