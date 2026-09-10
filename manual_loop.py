"""Raw Claude API agentic loop.

Deliberately not using the Agent SDK, so the underlying mechanism is
visible: inspect stop_reason, execute the requested tool(s), append results
to conversation history, and loop until stop_reason is "end_turn".

The safety-critical part of this project (the red-flag override) lives
entirely in tools.py, not here or in the system prompt — assess_symptoms
returns "er-now" for a red-flag symptom no matter what the model does with
that result, so a model that ignores or downplays the triage level still
can't change what was actually computed.

Run: export ANTHROPIC_API_KEY=... && python3 manual_loop.py
"""

import json

import anthropic

from tools import SessionState, assess_symptoms, escalate_to_care

MODEL = "claude-opus-5"

TOOLS = [
    {
        "name": "assess_symptoms",
        "description": (
            "Given a list of reported symptoms, return candidate conditions with "
            "confidence scores and a triage_level (self-care / see-a-doctor / "
            "er-now). Always call this before escalate_to_care. If triage_level "
            "comes back 'er-now', tell the user to seek emergency care immediately "
            "— do not soften or second-guess that result."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symptoms": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["symptoms"],
        },
    },
    {
        "name": "escalate_to_care",
        "description": (
            "Produce a self-contained handoff summary of the most recent "
            "assessment, for a human or downstream system with no access to this "
            "conversation. Requires assess_symptoms to have been called first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"notes": {"type": "string"}},
        },
    },
]

SYSTEM_PROMPT = (
    "You are a symptom-checking assistant. You are not a doctor and must never "
    "state or imply a confirmed diagnosis. Always call assess_symptoms before "
    "offering any guidance, and always relay its disclaimer and triage_level "
    "verbatim in your response. If triage_level is 'er-now', lead with that and "
    "tell the user to seek emergency care immediately."
)


def dispatch(state: SessionState, tool_name: str, tool_input: dict) -> dict:
    handlers = {
        "assess_symptoms": lambda: assess_symptoms(state, tool_input["symptoms"]),
        "escalate_to_care": lambda: escalate_to_care(state, tool_input.get("notes", "")),
    }
    return handlers[tool_name]()


def run(user_message: str) -> str:
    client = anthropic.Anthropic()
    state = SessionState()
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return next((b.text for b in response.content if b.type == "text"), "")

        if response.stop_reason != "tool_use":
            raise RuntimeError(f"Unhandled stop_reason: {response.stop_reason}")

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = dispatch(state, block.name, block.input)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                    "is_error": bool(result.get("isError")),
                }
            )
        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    print(run("I've had a runny nose, sore throat, and a mild cough for two days. What could this be?"))
