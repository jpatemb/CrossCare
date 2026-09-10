"""Claude Agent SDK bot wrapping tools.py.

manual_loop.py proves the mechanism works with a hand-rolled loop; this file
wires the same tools.py logic into an actual runnable Claude Agent SDK agent
and adds a second, independent safety layer on top of it.

Defense in depth, two layers, two different failure modes:

1. tools.py's assess_symptoms() deterministically computes triage_level from
   RED_FLAG_SYMPTOMS / the condition knowledge base. This guards against the
   *diagnostic logic* getting it wrong.
2. The PreToolUse hook below guards against a *different* failure: the model
   correctly receiving an "er-now" assess_symptoms result, then mis-relaying
   or softening it when it calls finalize_triage to deliver the answer to
   the user. The hook re-reads the session's actual last_assessment and
   denies finalize_triage outright if the model's proposed triage_level is
   less urgent than what was actually computed - before finalize_triage's own
   tool body ever runs.

Run: export ANTHROPIC_API_KEY=... && python3 agent_sdk_bot.py
"""

import asyncio
import json
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookContext,
    HookMatcher,
    TextBlock,
    ToolAnnotations,
    create_sdk_mcp_server,
    tool,
)

from tools import SessionState, assess_symptoms, escalate_to_care

MODEL = "claude-opus-5"
SERVER_NAME = "symptomchecker"

# Mirrors TRIAGE_LEVELS ordering in tools.py - higher is more urgent. Kept
# separate from tools.py because this is hook-layer policy (what counts as
# "at least as urgent as"), not diagnostic logic.
_TRIAGE_RANK = {"self-care": 0, "see-a-doctor": 1, "er-now": 2}

ASSESS_SYMPTOMS_SCHEMA = {
    "type": "object",
    "properties": {"symptoms": {"type": "array", "items": {"type": "string"}}},
    "required": ["symptoms"],
}

SYSTEM_PROMPT = (
    "You are a symptom-checking assistant. You are not a doctor and must never "
    "state or imply a confirmed diagnosis. Always call assess_symptoms before "
    "offering any guidance. Once you have an assessment, call finalize_triage "
    "exactly once to deliver your answer, passing through its triage_level and "
    "disclaimer verbatim - do not soften or second-guess an 'er-now' result. "
    "finalize_triage will be rejected if you understate the urgency."
)


def _as_tool_result(result: dict) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(result)}], "is_error": bool(result.get("isError"))}


def build_tools(state: SessionState) -> list:
    """Tool functions closing over this run's SessionState, so the SDK's
    in-process tool handlers and the PreToolUse hook below share the same
    assessment history.
    """

    @tool(
        "assess_symptoms",
        "Given a list of reported symptoms, return candidate conditions with "
        "confidence scores and a triage_level (self-care / see-a-doctor / "
        "er-now). Always call this before finalize_triage or escalate_to_care.",
        ASSESS_SYMPTOMS_SCHEMA,
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def _assess_symptoms(args: dict[str, Any]) -> dict[str, Any]:
        return _as_tool_result(assess_symptoms(state, args["symptoms"]))

    @tool(
        "escalate_to_care",
        "Produce a self-contained handoff summary of the most recent "
        "assessment, for a human or downstream system with no access to this "
        "conversation. Requires assess_symptoms to have been called first.",
        {"notes": str},
    )
    async def _escalate_to_care(args: dict[str, Any]) -> dict[str, Any]:
        return _as_tool_result(escalate_to_care(state, args.get("notes", "")))

    @tool(
        "finalize_triage",
        "Deliver the final triage result to the user. Call exactly once, "
        "after assess_symptoms, with the triage_level it returned and the "
        "message you want to show the user. A triage_level less urgent than "
        "what assess_symptoms actually computed will be rejected.",
        {"triage_level": str, "message": str},
    )
    async def _finalize_triage(args: dict[str, Any]) -> dict[str, Any]:
        return _as_tool_result({"isError": False, "delivered": args})

    return [_assess_symptoms, _escalate_to_care, _finalize_triage]


def build_red_flag_floor_hook(state: SessionState):
    """PreToolUse hook: independently enforces that finalize_triage can never
    report a lower-urgency triage_level than the session's actual last
    assess_symptoms result. Runs before finalize_triage's own tool body.
    """

    async def enforce_red_flag_floor(
        input_data: dict[str, Any], tool_use_id: str | None, context: HookContext
    ) -> dict[str, Any]:
        if input_data.get("tool_name") != f"mcp__{SERVER_NAME}__finalize_triage":
            return {}

        if state.last_assessment is None:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        "finalize_triage called before assess_symptoms; no assessment on record."
                    ),
                }
            }

        required = state.last_assessment["triage_level"]
        proposed = input_data.get("tool_input", {}).get("triage_level")
        if _TRIAGE_RANK.get(proposed, -1) < _TRIAGE_RANK.get(required, 0):
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"assess_symptoms computed triage_level={required!r}; finalize_triage "
                        f"may not report a lower urgency ({proposed!r}). Call finalize_triage "
                        f"again with triage_level={required!r}."
                    ),
                }
            }
        return {}

    return enforce_red_flag_floor


def build_options(state: SessionState) -> ClaudeAgentOptions:
    server = create_sdk_mcp_server(name=SERVER_NAME, version="1.0.0", tools=build_tools(state))
    tool_names = [
        f"mcp__{SERVER_NAME}__assess_symptoms",
        f"mcp__{SERVER_NAME}__escalate_to_care",
        f"mcp__{SERVER_NAME}__finalize_triage",
    ]
    return ClaudeAgentOptions(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={SERVER_NAME: server},
        allowed_tools=tool_names,
        hooks={
            "PreToolUse": [
                HookMatcher(
                    matcher=f"mcp__{SERVER_NAME}__finalize_triage",
                    hooks=[build_red_flag_floor_hook(state)],
                )
            ]
        },
    )


async def run(user_message: str) -> str:
    state = SessionState()
    options = build_options(state)

    texts: list[str] = []
    async with ClaudeSDKClient(options=options) as client:
        await client.query(user_message)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                texts.extend(block.text for block in message.content if isinstance(block, TextBlock))
    return "\n".join(texts)


if __name__ == "__main__":
    print(
        asyncio.run(
            run("I've had a runny nose, sore throat, and a mild cough for two days. What could this be?")
        )
    )
