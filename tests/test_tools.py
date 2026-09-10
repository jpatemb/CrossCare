from tools import DISCLAIMER, SessionState, assess_symptoms, escalate_to_care


def test_assess_requires_at_least_one_symptom():
    state = SessionState()
    result = assess_symptoms(state, [])
    assert result["isError"] is True
    assert result["errorCategory"] == "validation"


def test_assess_ignores_blank_and_whitespace_only_entries():
    state = SessionState()
    result = assess_symptoms(state, ["", "   "])
    assert result["isError"] is True
    assert result["errorCategory"] == "validation"


def test_common_cold_symptoms_trigger_self_care():
    state = SessionState()
    result = assess_symptoms(state, ["runny nose", "sore throat", "cough", "sneezing"])
    assert result["isError"] is False
    assert result["triage_level"] == "self-care"
    assert result["red_flags_detected"] == []


def test_appendicitis_symptoms_trigger_er_now_via_condition_match():
    """A condition can force er-now through its own baseline_triage, without
    any literal red-flag symptom being present."""
    state = SessionState()
    result = assess_symptoms(state, ["abdominal pain", "fever", "nausea", "loss of appetite"])
    assert result["isError"] is False
    assert result["triage_level"] == "er-now"
    assert result["red_flags_detected"] == []
    assert result["candidates"][0]["condition"] == "appendicitis"


def test_red_flag_forces_er_now_even_with_no_condition_match():
    """The red-flag override is independent of the candidate matcher: a red
    flag symptom with otherwise no matching condition still forces er-now."""
    state = SessionState()
    result = assess_symptoms(state, ["chest pain"])
    assert result["isError"] is False
    assert result["triage_level"] == "er-now"
    assert "chest pain" in result["red_flags_detected"]


def test_red_flag_overrides_a_low_urgency_candidate_match():
    """Mostly-cold symptoms plus one red flag must still resolve to er-now,
    not the cold's self-care baseline — this is the project's core guard."""
    state = SessionState()
    result = assess_symptoms(state, ["runny nose", "sore throat", "cough", "sneezing", "chest pain"])
    assert result["isError"] is False
    assert result["triage_level"] == "er-now"
    assert "chest pain" in result["red_flags_detected"]


def test_low_confidence_match_defaults_to_see_a_doctor_not_self_care():
    """A partial overlap with influenza's symptom set (2 of 5, 40%) clears
    the matching threshold but not the high-confidence one, so it should
    fall back to the cautious default rather than claiming self-care."""
    state = SessionState()
    result = assess_symptoms(state, ["fever", "body aches"])
    assert result["isError"] is False
    assert 0 < result["candidates"][0]["confidence"] < 0.5
    assert result["triage_level"] == "see-a-doctor"


def test_no_matching_condition_defaults_to_self_care():
    state = SessionState()
    result = assess_symptoms(state, ["hiccups"])
    assert result["isError"] is False
    assert result["triage_level"] == "self-care"
    assert result["candidates"] == []


def test_candidates_sorted_by_confidence_descending():
    state = SessionState()
    result = assess_symptoms(state, ["fever", "body aches", "fatigue", "cough", "chills"])
    confidences = [c["confidence"] for c in result["candidates"]]
    assert confidences == sorted(confidences, reverse=True)


def test_symptom_normalization_is_case_and_whitespace_insensitive():
    state = SessionState()
    result = assess_symptoms(state, ["  Chest Pain  "])
    assert result["triage_level"] == "er-now"
    assert "chest pain" in result["red_flags_detected"]


def test_disclaimer_present_on_every_successful_result():
    for symptoms in (["runny nose"], ["chest pain"], ["hiccups"]):
        result = assess_symptoms(SessionState(), symptoms)
        assert result["disclaimer"] == DISCLAIMER


def test_escalate_blocked_until_assessment_done():
    state = SessionState()
    result = escalate_to_care(state, notes="patient requests callback")
    assert result["isError"] is True
    assert result["errorCategory"] == "permission"


def test_escalate_returns_last_assessments_triage_level():
    state = SessionState()
    assess_symptoms(state, ["chest pain"])
    handoff = escalate_to_care(state, notes="patient reports 8/10 pain")
    assert handoff["isError"] is False
    assert handoff["triage_level"] == "er-now"
    assert handoff["notes"] == "patient reports 8/10 pain"
    assert handoff["disclaimer"] == DISCLAIMER


def test_escalate_reflects_most_recent_assessment_not_the_first():
    state = SessionState()
    assess_symptoms(state, ["chest pain"])
    assess_symptoms(state, ["runny nose", "sore throat", "cough", "sneezing"])
    handoff = escalate_to_care(state)
    assert handoff["triage_level"] == "self-care"
