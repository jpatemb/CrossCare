"""Business logic and mock knowledge base for CrossCare.

Pure Python, no Claude API calls here — this module is fully unit-testable
offline. manual_loop.py wraps these functions as tools for a live agent.

The core idea: a symptom-checking agent's confidence-scored condition
matching is inherently fuzzy, but its safety-critical decision (should this
person go to the ER right now?) shouldn't be. RED_FLAG_SYMPTOMS is a fixed,
hardcoded list checked in plain Python, deterministically forcing "er-now"
triage whenever present — independent of, and able to override, whatever
the fuzzier candidate-matching logic below it concludes. That separation
(deterministic safety guard in front of probabilistic reasoning) is the
whole point of this project, not the diagnostic accuracy of the mock
knowledge base itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DISCLAIMER = (
    "This is not a medical diagnosis and is not a substitute for professional "
    "medical advice, diagnosis, or treatment. If you are experiencing a medical "
    "emergency, call your local emergency number immediately."
)

# Hardcoded red-flag symptoms. If any of these is present, triage is forced
# to "er-now" no matter what the candidate-condition matching below produces.
RED_FLAG_SYMPTOMS = {
    "chest pain": "possible cardiac event",
    "pressure in chest": "possible cardiac event",
    "difficulty breathing": "airway or respiratory compromise",
    "shortness of breath at rest": "airway or respiratory compromise",
    "slurred speech": "possible stroke",
    "facial drooping": "possible stroke",
    "sudden numbness": "possible stroke",
    "sudden weakness on one side": "possible stroke",
    "severe bleeding": "uncontrolled hemorrhage",
    "uncontrolled bleeding": "uncontrolled hemorrhage",
}

TRIAGE_LEVELS = ("self-care", "see-a-doctor", "er-now")

# Mock condition knowledge base. Each condition's baseline_triage is only
# used when its confidence is the strongest match — it is independent of
# (and can be overridden by) a red-flag hit.
_CONDITIONS = [
    {
        "name": "common cold",
        "symptoms": {"runny nose", "sore throat", "cough", "sneezing", "mild fever"},
        "baseline_triage": "self-care",
    },
    {
        "name": "seasonal allergies",
        "symptoms": {"runny nose", "sneezing", "itchy eyes", "congestion"},
        "baseline_triage": "self-care",
    },
    {
        "name": "influenza",
        "symptoms": {"fever", "body aches", "fatigue", "cough", "chills"},
        "baseline_triage": "see-a-doctor",
    },
    {
        "name": "strep throat",
        "symptoms": {"sore throat", "fever", "swollen glands", "difficulty swallowing"},
        "baseline_triage": "see-a-doctor",
    },
    {
        "name": "migraine",
        "symptoms": {"headache", "nausea", "light sensitivity", "visual aura"},
        "baseline_triage": "see-a-doctor",
    },
    {
        "name": "urinary tract infection",
        "symptoms": {"burning urination", "frequent urination", "pelvic pain"},
        "baseline_triage": "see-a-doctor",
    },
    {
        "name": "appendicitis",
        "symptoms": {"abdominal pain", "fever", "nausea", "loss of appetite"},
        "baseline_triage": "er-now",
    },
    {
        "name": "myocardial infarction",
        "symptoms": {"chest pain", "shortness of breath at rest", "sweating", "nausea", "pain radiating to arm"},
        "baseline_triage": "er-now",
    },
]

# Only match candidates on at least this much symptom overlap with the
# condition's full symptom set, so a single incidental symptom doesn't
# surface an unrelated condition.
_MIN_MATCH_CONFIDENCE = 0.34
# Only adopt a candidate's baseline_triage directly when it's this confident;
# below that, fall back to a more conservative default (see assess_symptoms).
_HIGH_CONFIDENCE = 0.5


def _error(category: str, message: str, retryable: bool) -> dict:
    """Structured error shape: category + retryable flag, not a generic
    'operation failed' string, so a calling agent can react appropriately.

    category is one of "transient" | "validation" | "business" | "permission".
    """
    return {
        "isError": True,
        "errorCategory": category,
        "isRetryable": retryable,
        "message": message,
    }


@dataclass
class SessionState:
    """Tracks the most recent assessment this session, so escalate_to_care
    can be gated on assess_symptoms having run first.
    """

    last_assessment: dict | None = None
    events: list[str] = field(default_factory=list)


def _normalize(symptoms: list[str]) -> set[str]:
    return {s.strip().lower() for s in symptoms if s and s.strip()}


def assess_symptoms(state: SessionState, symptoms: list[str]) -> dict:
    """Structured intake -> candidate conditions with confidence scores, plus
    a triage urgency level. Every successful result carries `disclaimer`.
    """
    normalized = _normalize(symptoms)
    state.events.append(f"assess_symptoms({sorted(normalized)!r})")

    if not normalized:
        return _error("validation", "At least one symptom is required.", retryable=False)

    matched_red_flags = {s: RED_FLAG_SYMPTOMS[s] for s in normalized if s in RED_FLAG_SYMPTOMS}

    candidates = []
    for condition in _CONDITIONS:
        overlap = normalized & condition["symptoms"]
        if not overlap:
            continue
        confidence = round(len(overlap) / len(condition["symptoms"]), 2)
        if confidence < _MIN_MATCH_CONFIDENCE:
            continue
        candidates.append(
            {
                "condition": condition["name"],
                "confidence": confidence,
                "matched_symptoms": sorted(overlap),
                "_baseline_triage": condition["baseline_triage"],
            }
        )
    candidates.sort(key=lambda c: c["confidence"], reverse=True)

    if matched_red_flags:
        # Deterministic override: red flags win regardless of candidate
        # confidence, even if no condition in the knowledge base matched.
        triage_level = "er-now"
        triage_reason = (
            "Red-flag symptom(s) detected: "
            + ", ".join(f"{symptom} ({reason})" for symptom, reason in sorted(matched_red_flags.items()))
            + ". This overrides any other assessment."
        )
    elif candidates and candidates[0]["confidence"] >= _HIGH_CONFIDENCE:
        triage_level = candidates[0]["_baseline_triage"]
        triage_reason = (
            f"Based on strongest match: {candidates[0]['condition']} "
            f"({candidates[0]['confidence']:.0%} symptom overlap)."
        )
    elif candidates:
        # Low-confidence matches never get to claim a low-urgency triage on
        # their own; when unsure, default to the more cautious option.
        triage_level = "see-a-doctor"
        triage_reason = "Possible matches found but confidence is low; professional evaluation recommended."
    else:
        triage_level = "self-care"
        triage_reason = "No matching conditions in the knowledge base; monitor symptoms and rest."

    result = {
        "isError": False,
        "candidates": [
            {"condition": c["condition"], "confidence": c["confidence"], "matched_symptoms": c["matched_symptoms"]}
            for c in candidates
        ],
        "triage_level": triage_level,
        "triage_reason": triage_reason,
        "red_flags_detected": sorted(matched_red_flags),
        "disclaimer": DISCLAIMER,
    }
    state.last_assessment = result
    return result


def escalate_to_care(state: SessionState, notes: str = "") -> dict:
    """Structured handoff summarizing the most recent assessment. Requires
    assess_symptoms to have run first — a human or downstream system reading
    this has no access to the conversation transcript, so it must be
    self-contained.
    """
    if state.last_assessment is None:
        return _error("permission", "No assessment on record. Call assess_symptoms first.", retryable=False)

    state.events.append("escalate_to_care(...)")
    return {
        "isError": False,
        "triage_level": state.last_assessment["triage_level"],
        "candidates": state.last_assessment["candidates"],
        "notes": notes,
        "disclaimer": DISCLAIMER,
        "status": "escalated",
    }
