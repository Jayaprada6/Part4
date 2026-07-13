import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import requests
from jsonschema import validate, ValidationError

from dotenv import load_dotenv
import os

load_dotenv()

LLM_API_KEY=os.getenv("API_KEY")
LLM_API_URL="https://openrouter.ai/api/v1/chat/completions" 
LLM_MODEL="openai/gpt-4o-mini"  

MODEL_PATH="best_model.pkl"

if LLM_API_KEY is None:
    raise ValueError("API_KEY not found in .env file")


FEATURE_ORDER=[
    "Priority_level",
    "Severity_level",
    "Agent_age",
    "ticket_month",
    "ticket_dayofweek",
    "agent_assigned",
    "Request Category_Login Access",
    "Request Category_Software",
    "Request Category_System",
    "Issue Type_IT Request",
]

PRIORITY_MAP = {0: "Unassigned", 1: "Low", 2: "Mid", 3: "High"}
SEVERITY_MAP = {0: "Unclassified", 1: "Minor", 2: "Normal", 3: "Mayor", 4: "Urgent"}

def call_llm(system_prompt: str, user_prompt: str, temperature: float=0.0, max_tokens: int=512):
    
    payload={
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers={
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    response=requests.post(LLM_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        print(f"LLM call failed with status {response.status_code}: {response.text}")
        return None

    return response.json()["choices"][0]["message"]["content"]


# Demonstration of call_llm with a simple test prompt
print("=== call_llm smoke test ===")
test_output=call_llm(
    system_prompt="You are a helpful assistant.",
    user_prompt="Reply with only the word: hello",
    temperature=0.0,
    max_tokens=10,
)
print("Test output:", test_output)
print()

def has_pii(text: str) -> bool:
    email_pattern=r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    phone_pattern=r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))


def guarded_call_llm(system_prompt: str, user_prompt: str, temperature: float=0.0, max_tokens: int=512):
    """Wraps call_llm with the PII guardrail. Blocks the call if PII is found."""
    if has_pii(user_prompt):
        print("Input blocked: PII detected.")
        return None
    return call_llm(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)


print("=== PII guardrail demonstration ===")
pii_input="Contact the requester at jane.doe@example.com for more info."
clean_input="The requester's ticket was reopened after a failed login attempt."

print("Blocked-input test:")
_ = guarded_call_llm("You are a helpful assistant.", pii_input)

print("\nClean-input test (should proceed to LLM call):")
_ = guarded_call_llm("You are a helpful assistant.", clean_input, max_tokens=10)
print()


print("=== Loading model ===")
model=joblib.load(MODEL_PATH)
print("Model loaded:", type(model))
print()


def encode_record(features: dict) -> "pd.DataFrame":
    """
    Takes a hand-crafted feature dict with human-readable keys and returns a
    1-row DataFrame with columns matching FEATURE_ORDER, ready for
    model.predict(). Returned as a DataFrame (not a bare ndarray) so the
    column names line up with what the pipeline was fitted on.

    Expected input keys:
      Priority_level (int 0-3), Severity_level (int 0-4), Agent_age (int),
      Ticket Date (str 'YYYY-MM-DD'), Agent ID (int or None),
      Request Category (one of Hardware/Login Access/Software/System),
      Issue Type (one of 'IT Error'/'IT Request')
    """
    ticket_date=pd.to_datetime(features["Ticket Date"])
    ticket_month=ticket_date.month
    ticket_dayofweek=ticket_date.dayofweek

    agent_assigned=1 if features.get("Agent ID") not in (None, "", 0) else 0

    request_category=features["Request Category"]
    issue_type=features["Issue Type"]

    row={
        "Priority_level": features["Priority_level"],
        "Severity_level": features["Severity_level"],
        "Agent_age": features["Agent_age"],
        "ticket_month": ticket_month,
        "ticket_dayofweek": ticket_dayofweek,
        "agent_assigned": agent_assigned,
        "Request Category_Login Access": 1 if request_category == "Login Access" else 0,
        "Request Category_Software": 1 if request_category == "Software" else 0,
        "Request Category_System": 1 if request_category == "System" else 0,
        "Issue Type_IT Request": 1 if issue_type == "IT Request" else 0,
    }
    return pd.DataFrame([[row[col] for col in FEATURE_ORDER]], columns=FEATURE_ORDER)


TEST_INPUTS=[
    {
        "Priority_level": 3,
        "Severity_level": 4,
        "Agent_age": 29,
        "Ticket Date": "2020-06-15",
        "Agent ID": 12,
        "Request Category": "System",
        "Issue Type": "IT Request",
    },
    {
        "Priority_level": 0,
        "Severity_level": 1,
        "Agent_age": 45,
        "Ticket Date": "2019-03-02",
        "Agent ID": None,
        "Request Category": "Hardware",
        "Issue Type": "IT Error",
    },
    {
        "Priority_level": 1,
        "Severity_level": 2,
        "Agent_age": 34,
        "Ticket Date": "2021-11-09",
        "Agent ID": 47,
        "Request Category": "Login Access",
        "Issue Type": "IT Request",
    },
]


EXPLANATION_SCHEMA={
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"},
    },
    "required": [
        "prediction_label",
        "confidence_level",
        "top_reason",
        "second_reason",
        "next_step",
    ],
}

FALLBACK_EXPLANATION={
    "prediction_label": None,
    "confidence_level": None,
    "top_reason": None,
    "second_reason": None,
    "next_step": None,
}


SYSTEM_PROMPT = """You are an assistant that explains machine learning predictions to IT support managers who are not data scientists.
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
Do not include any text before or after the JSON object."""

USER_PROMPT_TEMPLATE="""Feature values:
{feature_values}

Predicted class: {predicted_class}
Predicted probability: {predicted_probability}

Explain this prediction as a JSON object following the required schema."""


def build_user_prompt(features: dict, predicted_class, predicted_probability: float) -> str:
    readable_features={
        "Priority": PRIORITY_MAP.get(features["Priority_level"], features["Priority_level"]),
        "Severity": SEVERITY_MAP.get(features["Severity_level"], features["Severity_level"]),
        "Agent age": features["Agent_age"],
        "Ticket date": features["Ticket Date"],
        "Agent assigned": features.get("Agent ID") is not None,
        "Request category": features["Request Category"],
        "Issue type": features["Issue Type"],
    }
    return USER_PROMPT_TEMPLATE.format(
        feature_values=json.dumps(readable_features, indent=2),
        predicted_class=predicted_class,
        predicted_probability=round(float(predicted_probability), 4),
    )


def get_validated_explanation(system_prompt: str, user_prompt: str, temperature: float = 0.0):
    """Calls the LLM, parses the JSON response, and validates it against
    EXPLANATION_SCHEMA. Returns (parsed_dict_or_fallback, raw_response, status)."""

    if has_pii(user_prompt):
        print("Input blocked: PII detected.")
        return dict(FALLBACK_EXPLANATION), None, "blocked (PII)"

    raw_response=call_llm(system_prompt, user_prompt, temperature=temperature)
    if raw_response is None:
        return dict(FALLBACK_EXPLANATION), None, "fail (no response)"

    try:
        parsed=json.loads(raw_response.strip())
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return dict(FALLBACK_EXPLANATION), raw_response, f"fail (JSONDecodeError: {e})"

    try:
        validate(instance=parsed, schema=EXPLANATION_SCHEMA)
    except ValidationError as e:
        print(f"Schema validation error: {e.message}")
        return dict(FALLBACK_EXPLANATION), raw_response, f"fail (ValidationError: {e.message})"

    return parsed, raw_response, "pass"


print("=== Predict + explain pipeline (temperature=0.0) ===")
results_temp0 = []

for i, features in enumerate(TEST_INPUTS, start=1):
    encoded= encode_record(features)
    predicted_class= model.predict(encoded)[0]
    predicted_proba= model.predict_proba(encoded)[0]
    predicted_probability= predicted_proba[list(model.classes_).index(predicted_class)]

    user_prompt= build_user_prompt(features, predicted_class, predicted_probability)
    parsed, raw, status= get_validated_explanation(SYSTEM_PROMPT, user_prompt, temperature=0.0)

    print(f"\n--- Input {i} ---")
    print("Features:", features)
    print("Predicted class:", predicted_class)
    print("Predicted probability:", round(float(predicted_probability), 4))
    print("Raw LLM response:", raw)
    print("Validation outcome:", status)
    print("Parsed explanation:", parsed)

    results_temp0.append({
        "features": features,
        "predicted_class": predicted_class,
        "predicted_probability": predicted_probability,
        "raw_response": raw,
        "status": status,
        "explanation": parsed,
    })


print("\n=== Temperature A/B comparison (temp=0.0 vs temp=0.7) ===")
results_temp07=[]

for i,features in enumerate(TEST_INPUTS, start=1):
    encoded = encode_record(features)
    predicted_class = model.predict(encoded)[0]
    predicted_proba = model.predict_proba(encoded)[0]
    predicted_probability = predicted_proba[list(model.classes_).index(predicted_class)]

    user_prompt = build_user_prompt(features, predicted_class, predicted_probability)
    parsed, raw, status = get_validated_explanation(SYSTEM_PROMPT, user_prompt, temperature=0.7)

    print(f"\n--- Input {i} (temp=0.7) ---")
    print("Raw LLM response:", raw)
    print("Validation outcome:", status)

    results_temp07.append({
        "features": features,
        "raw_response": raw,
        "status": status,
        "explanation": parsed,
    })

print("\nDone. See README.md for the write-up, tables, and prompt rationale.")
