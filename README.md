# CrossCare

A tool-calling symptom-checker agent. Given structured symptom intake, it
returns candidate conditions with confidence scores and a triage urgency
level (`self-care` / `see-a-doctor` / `er-now`).

The interesting part isn't the mock diagnostic knowledge base — it's the
safety pattern: a fixed set of red-flag symptoms (chest pain, stroke signs,
breathing/bleeding emergencies) is checked in plain Python and
**deterministically forces `er-now`**, overriding whatever the fuzzier
confidence-scored condition matching — or a live model call — concludes.
Every successful result also carries a fixed
"not a substitute for professional medical advice" disclaimer. Started as a
hands-on project for studying agent architecture (tool design, deterministic
safety guards in front of probabilistic reasoning, structured handoffs).

**This is not a medical device and does not give medical advice.** It's a
learning project with a mock, hand-written condition list — nowhere close
to clinically validated.

## Structure

| File | What it does |
|---|---|
| `tools.py` | The triage logic and mock condition knowledge base. `RED_FLAG_SYMPTOMS` is the deterministic override; `assess_symptoms` / `escalate_to_care` are pure, offline-testable functions with no API calls. |
| `manual_loop.py` | A raw Claude API agentic loop (`stop_reason` inspection, tool dispatch, message loop) that wraps `tools.py` as tools for a live agent. |
| `agent_sdk_bot.py` | The same agent built on the real Claude Agent SDK instead of a hand-rolled loop: `tools.py`'s functions become SDK MCP tools via `@tool`/`create_sdk_mcp_server`, plus a third tool, `finalize_triage`, that the model must call to deliver its answer. A `PreToolUse` hook on `finalize_triage` (`build_red_flag_floor_hook`) is a *second, independent* safety layer — it re-reads the session's actual computed `triage_level` and denies the call if the model's proposed level is less urgent, catching the model mis-relaying a correct result rather than `tools.py` computing a wrong one. |
| `conftest.py` | Offline test guards — strips `ANTHROPIC_API_KEY` and blocks outbound sockets for every test, so the suite behaves the same with or without a key. |
| `tests/conftest.py` | Puts the repo root on `sys.path` for `tests/` (this repo isn't a package, since `manual_loop.py` needs a flat `from tools import ...`). |
| `tests/test_tools.py` | Offline tests covering the red-flag override (including when it must beat a low-urgency candidate match), triage-confidence thresholds, the `assess_symptoms` → `escalate_to_care` prerequisite gate, and the disclaimer contract. |
| `tests/test_agent_sdk_bot.py` | Offline tests for the `PreToolUse` hook's decision logic — pure async functions, dicts in/dicts out, no CLI subprocess or network — covering the deny/allow cases including a model trying to downgrade an `er-now` result to `self-care`. |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
```

## Run

Offline (no API key needed) — verifies the triage logic itself:

```bash
pytest tests/ -v
```

`conftest.py` strips `ANTHROPIC_API_KEY` and blocks outbound sockets for
every test automatically, so the `export` above cannot contaminate a test
run — the suite behaves the same on a machine with no key at all.

Live, raw API loop:

```bash
python3 manual_loop.py
```

Live, Agent SDK bot (requires the `claude-agent-sdk` package, which shells
out to the Claude Code CLI):

```bash
python3 agent_sdk_bot.py
```

The default prompt reports cold symptoms (self-care). Try editing it to
include `"chest pain"` alongside otherwise mild symptoms, and watch the
triage level come back `er-now` regardless — that's `RED_FLAG_SYMPTOMS`
overriding the candidate match, not the model deciding to be cautious. In
`agent_sdk_bot.py` you can additionally test the second safety layer: patch
`finalize_triage_tool` (or just watch the hook's deny reason in stderr) to
confirm a model call that tries to soften an `er-now` result gets rejected
before it ever reaches the user.

## What to look for while testing

- In `tools.py`, comment out the `if matched_red_flags:` branch in
  `assess_symptoms` and re-run the red-flag tests — they should fail. The
  override is a separate code path, not an emergent property of the
  confidence scoring, and losing it should be loud (a failing test), not
  silent.
- Compare `test_appendicitis_symptoms_trigger_er_now_via_condition_match`
  with `test_red_flag_forces_er_now_even_with_no_condition_match` — two
  different ways to reach `er-now`: a confident condition match with its
  own high-urgency baseline, versus the red-flag override firing with *no*
  condition match at all.
- Try `manual_loop.py` with a prompt that includes a red-flag symptom and
  see whether the model relays `triage_level: "er-now"` faithfully per the
  system prompt's instruction, or whether it hedges.
- In `agent_sdk_bot.py`, the red-flag override exists at two independent
  layers: `tools.py`'s `RED_FLAG_SYMPTOMS` check (same as above) computes
  the correct `triage_level`, and `build_red_flag_floor_hook`'s
  `PreToolUse` hook separately verifies that whatever the model passes to
  `finalize_triage` isn't a downgrade from that computed value. Comment out
  either one independently and the corresponding tests should fail — they
  guard against different failure modes and neither can substitute for the
  other.
