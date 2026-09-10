"""Claude Agent SDK bot for CrossCare, with a second, independent red-flag guard.

manual_loop.py's raw API loop relies on tools.py's assess_symptoms to compute
the correct triage_level, and on the model relaying that value verbatim. That
covers one failure mode (tools.py computing the wrong result would be caught
by tests/test_tools.py) but not another: the model calling finalize_triage
with a triage_level that doesn't match what assess_symptoms actually
returned (mis-relaying a correct result).

This bot adds a PreToolUse hook on finalize_triage that independently
re-reads the session's last assess_symptoms result and denies the tool call
outright if the model's proposed triage_level is less urgent than what was
actually computed. Same defense-in-depth pattern as the support-agent
project: a deterministic Python check sitting in front of the model's own
tool call, not just inside the tool the model calls voluntarily.

Run: export ANTHROPIC_API_KEY=... && python3 agent_sdk_bot.py
"""

import anyio

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    HookMatcher,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
    tool,
)

from tools import SessionState, TRIAGE_LEVELS, assess_symptoms, escalate_to_care

MODEL = "claude-opus-5"
SERVER_NAME = "crosscare"
FINALIZE_TOOL_WIRE_NAME = f"mcp__{SERVER_NAME}__finalize_triage"

SYSTEM_PROMPT = (
    "You are a symptom-checking assistant. You are not a doctor and must never "
    "state or imply a confirmed diagnosis. Always call assess_symptoms before "
    "offering any guidance. Once you have an assessment, you must deliver your "
    "final answer to the user by calling finalize_triage with the triage_level "
    "assess_symptoms returned (relayed verbatim, never softened or "
    "second-guessed) and a message that includes the disclaimer. If "
    "triage_level is 'er-now', lead the message with that and tell the user "
    "to seek emergency care immediately."
)


def _triage_rank(level: str) -> int:
    """Index into TRIAGE_LEVELS ('self-care' < 'see-a-doctor' < 'er-now'), so
    two triage levels can be compared for urgency, not just equality.
    """
    return TRIAGE_LEVELS.index(level)


def build_tools(state: SessionState):
    """Build the three SDK MCP tools bound to a single session's state.

    finalize_triage is not defined in tools.py: it's the delivery mechanism
    the model must go through to answer, which is what gives the PreToolUse
    hook below something to intercept before the answer reaches the user.
    """

    @tool(
        "assess_symptoms",
        (
            "Given a list of reported symptoms, return candidate conditions with "
            "confidence scores and a triage_level (self-care / see-a-doctor / "
            "er-now). Always call this before finalize_triage or escalate_to_care."
        ),
        {"symptoms": list[str]},
    )
    async def assess_symptoms_tool(args: dict) -> dict:
        result = assess_symptoms(state, args["symptoms"])
        return {"content": [{"type": "text", "text": str(result)}], "is_error": bool(result.get("isError"))}

    @tool(
        "escalate_to_care",
        (
            "Produce a self-contained handoff summary of the most recent "
            "assessment. Requires assess_symptoms to have been called first."
        ),
        {"notes": str},
    )
    async def escalate_to_care_tool(args: dict) -> dict:
        result = escalate_to_care(state, args.get("notes", ""))
        return {"content": [{"type": "text", "text": str(result)}], "is_error": bool(result.get("isError"))}

    @tool(
        "finalize_triage",
        (
            "Deliver the final answer to the user. triage_level must be the "
            "value assess_symptoms actually returned, relayed verbatim. Must "
            "be called exactly once, after assess_symptoms."
        ),
        {"triage_level": str, "message": str},
    )
    async def finalize_triage_tool(args: dict) -> dict:
        # No safety logic here on purpose: this tool just delivers whatever the
        # model asks it to. The independent check lives entirely in the
        # PreToolUse hook below, which runs before this handler ever executes.
        return {"content": [{"type": "text", "text": args["message"]}]}

    return [assess_symptoms_tool, escalate_to_care_tool, finalize_triage_tool]


def build_red_flag_floor_hook(state: SessionState):
    """PreToolUse hook: an independent floor under finalize_triage.

    Re-derives the minimum acceptable triage_level from state.last_assessment
    (populated by tools.py's own assess_symptoms, not by anything the model
    said) and denies the call if the model's proposed triage_level is less
    urgent. This is a second, independent check on the same invariant tools.py
    enforces internally -- it catches the model mis-relaying a correct
    assess_symptoms result, a failure mode tools.py's own logic can't see
    because by the time finalize_triage runs, tools.py's job is already done.
    """

    async def red_flag_floor(input_data, tool_use_id, context):
        if input_data.get("tool_name") != FINALIZE_TOOL_WIRE_NAME:
            return {}

        last_assessment = state.last_assessment
        if last_assessment is None:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        "finalize_triage called before assess_symptoms; no "
                        "assessment on record to verify against."
                    ),
                }
            }

        required_level = last_assessment["triage_level"]
        proposed_level = input_data.get("tool_input", {}).get("triage_level")

        if proposed_level not in TRIAGE_LEVELS:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"triage_level {proposed_level!r} is not one of {TRIAGE_LEVELS}."
                    ),
                }
            }

        if _triage_rank(proposed_level) < _triage_rank(required_level):
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"assess_symptoms computed triage_level={required_level!r}, "
                        f"but finalize_triage was called with the less urgent "
                        f"{proposed_level!r}. Call finalize_triage again with the "
                        "correct triage_level."
                    ),
                }
            }

        return {}

    return red_flag_floor


async def run(user_message: str) -> str:
    state = SessionState()
    server = create_sdk_mcp_server(name=SERVER_NAME, tools=build_tools(state))

    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={SERVER_NAME: server},
        allowed_tools=[
            f"mcp__{SERVER_NAME}__assess_symptoms",
            f"mcp__{SERVER_NAME}__escalate_to_care",
            FINALIZE_TOOL_WIRE_NAME,
        ],
        hooks={"PreToolUse": [HookMatcher(matcher=FINALIZE_TOOL_WIRE_NAME, hooks=[build_red_flag_floor_hook(state)])]},
    )

    final_text = ""
    async for message in query(prompt=user_message, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock) and block.name == FINALIZE_TOOL_WIRE_NAME:
                    final_text = block.input.get("message", "")
                elif isinstance(block, TextBlock):
                    final_text = final_text or block.text
        elif isinstance(message, ResultMessage) and message.is_error:
            raise RuntimeError(f"Query ended in error: {message.result}")

    return final_text


if __name__ == "__main__":
    print(anyio.run(run, "I've had a runny nose, sore throat, and a mild cough for two days. What could this be?"))
