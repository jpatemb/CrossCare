"""Offline tests for the PreToolUse red-flag floor hook.

Pure Python: dicts in, dict out (async, but no network and no CLI
subprocess), so these fit the project's offline-testable pattern with no
new mocking. Coverage here is for the hook's own decision logic, not
tool-calling behavior in general -- that already lives in
tests/test_tools.py.
"""

import asyncio

from agent_sdk_bot import FINALIZE_TOOL_WIRE_NAME, build_red_flag_floor_hook
from tools import SessionState, assess_symptoms


def _run_hook(hook, tool_input: dict) -> dict:
    input_data = {
        "hook_event_name": "PreToolUse",
        "tool_name": FINALIZE_TOOL_WIRE_NAME,
        "tool_input": tool_input,
        "tool_use_id": "toolu_test",
        "session_id": "sess_test",
        "transcript_path": "",
        "cwd": "",
    }
    return asyncio.run(hook(input_data, "toolu_test", {"signal": None}))


def test_ignores_calls_to_other_tools():
    state = SessionState()
    hook = build_red_flag_floor_hook(state)
    input_data = {
        "hook_event_name": "PreToolUse",
        "tool_name": "mcp__crosscare__assess_symptoms",
        "tool_input": {},
        "tool_use_id": "toolu_test",
        "session_id": "sess_test",
        "transcript_path": "",
        "cwd": "",
    }
    result = asyncio.run(hook(input_data, "toolu_test", {"signal": None}))
    assert result == {}


def test_denies_finalize_before_any_assessment():
    state = SessionState()
    hook = build_red_flag_floor_hook(state)
    result = _run_hook(hook, {"triage_level": "self-care", "message": "hi"})
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "no assessment" in result["hookSpecificOutput"]["permissionDecisionReason"].lower()


def test_denies_invalid_triage_level():
    state = SessionState()
    assess_symptoms(state, ["runny nose"])
    hook = build_red_flag_floor_hook(state)
    result = _run_hook(hook, {"triage_level": "probably-fine", "message": "hi"})
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_allows_finalize_that_matches_computed_triage_level():
    state = SessionState()
    assess_symptoms(state, ["runny nose", "sore throat", "cough", "sneezing"])
    hook = build_red_flag_floor_hook(state)
    result = _run_hook(hook, {"triage_level": "self-care", "message": "hi"})
    assert result == {}


def test_allows_finalize_that_matches_er_now():
    state = SessionState()
    assess_symptoms(state, ["chest pain"])
    hook = build_red_flag_floor_hook(state)
    result = _run_hook(hook, {"triage_level": "er-now", "message": "hi"})
    assert result == {}


def test_denies_model_downgrading_er_now_to_self_care():
    """The core guard: a red-flag symptom computes er-now in tools.py, but
    the model tries to soften it when delivering the answer. The hook must
    catch this independently of tools.py, which already did its job
    correctly by the time finalize_triage runs.
    """
    state = SessionState()
    assess_symptoms(state, ["chest pain"])
    hook = build_red_flag_floor_hook(state)
    result = _run_hook(hook, {"triage_level": "self-care", "message": "you'll be fine"})
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "er-now" in result["hookSpecificOutput"]["permissionDecisionReason"]
    assert "self-care" in result["hookSpecificOutput"]["permissionDecisionReason"]


def test_denies_model_downgrading_see_a_doctor_to_self_care():
    state = SessionState()
    assess_symptoms(state, ["fever", "body aches"])  # low-confidence -> see-a-doctor
    hook = build_red_flag_floor_hook(state)
    result = _run_hook(hook, {"triage_level": "self-care", "message": "rest up"})
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_allows_finalize_that_upgrades_beyond_computed_triage_level():
    """The floor only blocks understating urgency; a model being *more*
    cautious than assess_symptoms is not a safety problem."""
    state = SessionState()
    assess_symptoms(state, ["runny nose", "sore throat", "cough", "sneezing"])
    hook = build_red_flag_floor_hook(state)
    result = _run_hook(hook, {"triage_level": "er-now", "message": "just in case"})
    assert result == {}


def test_uses_most_recent_assessment_not_the_first():
    state = SessionState()
    assess_symptoms(state, ["chest pain"])
    assess_symptoms(state, ["runny nose", "sore throat", "cough", "sneezing"])
    hook = build_red_flag_floor_hook(state)
    result = _run_hook(hook, {"triage_level": "self-care", "message": "hi"})
    assert result == {}
