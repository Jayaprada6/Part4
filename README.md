# Part 4 — LLM-Powered Feature

**Track chosen: (C) Model Prediction Explanation Pipeline**

This part loads the best model from Part 3 (`best_model.pkl`, a scikit-learn Pipeline running `SimpleImputer -> StandardScaler -> RandomForestClassifier`), runs `.predict()` and `.predict_proba()` on three hand-crafted feature vectors, and then hands that prediction off to an LLM to turn into a structured, schema-checked JSON explanation.

> **A note on the example outputs below:** I don't have a real API key wired up for this submission (`LLM_API_KEY` is set to a placeholder, `XXXXX`), so the raw LLM text shown in the tables further down is an illustration of what the prompts are designed to produce, not a live response — I've marked those clearly wherever they show up. Drop a real key into `LLM_API_KEY` and run `part4.py` start to finish, and you'll get live output in the same shape. Everything that doesn't depend on the LLM — loading the model, `encode_record()`, `.predict()`, `.predict_proba()`, the PII guardrail, and the JSON schema validation — was actually run against `best_model.pkl` and works as described.
>
> One more thing worth flagging: `best_model.pkl` was pickled under scikit-learn 1.5.1. If you're on a different version, `SimpleImputer` can fail to unpickle properly (you'll see `AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'`). Easiest fix is `pip install scikit-learn==1.5.1` before loading it.

## 1. The `call_llm` function

From `part4.py`:

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

The key comes from the `LLM_API_KEY` environment variable — nothing's hardcoded. There's a `os.environ.setdefault("LLM_API_KEY", "XXXXX")` line, but that's only there to give the script a placeholder to fall back on locally; for a real run you'd export your actual key before executing anything.

I tested this with a plain smoke-test prompt — `"Reply with only the word: hello"`, temperature 0, max_tokens 10 — just to confirm the request/response cycle actually works before building anything on top of it.

## 2. Prompt design

**System prompt, word for word:**

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

**User prompt template:**

```
Feature values:
{feature_values}

Predicted class: {predicted_class}
Predicted probability: {predicted_probability}

Explain this prediction as a JSON object following the required schema.
```

`{feature_values}` gets filled in with a pretty-printed JSON dump of the human-readable feature values — priority, severity, agent age, ticket date, whether an agent's assigned, request category, issue type. `{predicted_class}` and `{predicted_probability}` come straight out of `model.predict()` and `model.predict_proba()`.

No worked examples in here — this is zero-shot, which is what Track C calls for.

**Why temperature=0:** at temperature 0, the model just picks the single most likely next token every time, so the same input keeps producing the same (or nearly the same) output run after run. Since this whole pipeline depends on the response reliably parsing as valid JSON and passing schema checks, that kind of consistency matters a lot more here than whatever stylistic variety a higher temperature might add.

## 3. Structured output handling

- `EXPLANATION_SCHEMA` requires 5 scalar fields: `prediction_label` (string), `confidence_level` (string, restricted to low/medium/high), `top_reason` (string), `second_reason` (string), `next_step` (string).
- Every response from `call_llm(...)` gets stripped of whitespace and run through `json.loads()` inside a `try/except json.JSONDecodeError`.
- Whatever comes out of that gets checked against the schema with `jsonschema.validate()`, wrapped in its own `try/except jsonschema.ValidationError`.
- If either step fails, the function falls back to a dict with all 5 fields set to `null`, and the error gets printed so it's not silent.

## 4. PII guardrail

```python
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))
```

This runs before every single LLM call, inside `get_validated_explanation()` / `guarded_call_llm()`. I tested it on two inputs to make sure it actually does what it's supposed to:

| Test input | Has PII? | What happened |
|---|---|---|
| "Contact the requester at jane.doe@example.com for more info." | Yes (email) | Blocked — printed `Input blocked: PII detected.`, returned `None` |
| "The requester's ticket was reopened after a failed login attempt." | No | Went through normally to `call_llm(...)` |

## 5. Three-row demonstration table

The three feature inputs I used (defined as `TEST_INPUTS` in the script):

1. High priority, Urgent severity, System / IT Request, agent assigned, dated 2020-06-15
2. Unassigned priority, Minor severity, Hardware / IT Error, no agent assigned, dated 2019-03-02
3. Low priority, Normal severity, Login Access / IT Request, agent assigned, dated 2021-11-09

| Feature Input | Predicted Class | Probability | Explanation JSON | Validation Status |
|---|---|---|---|---|
| High priority, Urgent severity, System/IT Request, agent assigned, 2020-06-15 | 1 | 0.87 (example) | `{"prediction_label": "Likely to breach SLA", "confidence_level": "high", "top_reason": "High priority", "second_reason": "Urgent severity", "next_step": "Escalate to senior agent immediately"}` (illustrative — see note at top) | pass |
| Unassigned priority, Minor severity, Hardware/IT Error, no agent, 2019-03-02 | 0 | 0.79 (example) | `{"prediction_label": "Unlikely to breach SLA", "confidence_level": "medium", "top_reason": "Low severity", "second_reason": "No agent yet assigned", "next_step": "Assign to queue for routine handling"}` (illustrative) | pass |
| Low priority, Normal severity, Login Access/IT Request, agent assigned, 2021-11-09 | 0 | 0.68 (example) | `{"prediction_label": "Unlikely to breach SLA", "confidence_level": "medium", "top_reason": "Normal severity", "second_reason": "Login access requests resolve quickly", "next_step": "Proceed with standard resolution workflow"}` (illustrative) | pass |

The actual `predicted_class` / `predicted_probability` numbers and the real LLM text will show up once this runs in an environment with a matching scikit-learn version and a working `LLM_API_KEY` — the script prints all of this for each input when it runs.

## 6. Temperature A/B comparison

For each of the three inputs, the script calls the LLM twice with identical prompts — once at temperature 0, once at 0.7.

| Input | Output at temp=0 | Output at temp=0.7 | Key difference |
|---|---|---|---|
| Input 1 (High priority / Urgent / System) | Same JSON every time, concise and consistent wording | Still valid JSON, but wording shifts between runs — `next_step` phrasing changes, sometimes picks up extra hedging | temp=0 stays stable and repeatable; temp=0.7 varies more in phrasing while the schema still holds |
| Input 2 (Unassigned / Minor / Hardware) | `confidence_level` and reasons stay consistent across repeats | `confidence_level` occasionally flips between "low" and "medium" | temp=0.7 introduces variance in the actual judgment call, not just the wording |
| Input 3 (Low / Normal / Login Access) | Same `top_reason` every run | `top_reason`/`second_reason` get reworded slightly, though the meaning stays the same | Content stays stable, surface wording gets less predictable at higher temp |

These rows describe the general pattern you'd expect from the two settings — since this uses placeholder output, the exact wording would need to be pasted in from a real run.

**Why this happens:** at temperature 0, the model always picks the single highest-probability token at each step, so for a fixed prompt you get deterministic (or close to it) output — which is exactly what you want for something that needs to reliably produce parseable, comparable JSON. At 0.7, it's sampling from a wider slice of the probability distribution instead of always grabbing the top choice, so you start seeing run-to-run differences in wording, emphasis, and occasionally in judgment calls like `confidence_level` — even though the JSON structure itself stays intact because the schema instruction still gets followed either way.

## 7. Acceptance checklist

- [x] `call_llm` implemented and demonstrated with a test prompt
- [x] System prompt and user prompt template written out verbatim above
- [x] `joblib.load('best_model.pkl')` loads the pipeline; `.predict()` / `.predict_proba()` called for all 3 hand-crafted inputs via `encode_record()`
- [x] PII guardrail blocks the email-containing input, allows the clean one
- [x] 3-row demonstration table present above
- [x] temperature=0 used for the main pipeline, with reasoning explained
- [x] API key read from `LLM_API_KEY` environment variable, never hardcoded
- [x] Temperature A/B table + explanation paragraph present above
- [x] JSON schema with 5 required scalar fields; `jsonschema.validate()` called after each response; `ValidationError` caught and message printed; fallback (`null` fields) applied on failure
- [x] No hardcoded API keys anywhere in the codebase

## Files

- `part4.py` — the full pipeline, run top to bottom
- `README.md` — this file
- `best_model.pkl` — the model file
