"""Synthetic controls for the public questionnaire adapter, with no patient data."""
import copy
import csv
import json

import pytest

from app.chorus_review import (
    ABSTAIN, POSITIVE, PROBABLE, PHENOTYPES, audit_violations,
    prepare_cases, run_case, main,
)


def instrument():
    specs = {}

    def field(name, kind="text", choices=""):
        specs[name] = {"Variable / Field Name": name, "Field Type": kind,
                       "Field Label": f"Synthetic definition of {name}",
                       "Choices, Calculations, OR Slider Labels": choices}

    for _, primary, _ in PHENOTYPES[:-1]:
        field(primary, "radio", "1, Yes | 2, No | 3, Uncertain")
    for name in ["sep3_v3", "sep5_v3", "sep7_v3", "sep_day_v3", "adhf2_v3", "adhf3_v3", "adhf6_v3",
                 "del2_camicu_v3", "del3_icdsc_v3", "shock1_v3"]:
        field(name)
    field("sep4_v3", "yesno")
    field("aki1_v3", "yesno")
    field("shock2_v4_sep", "radio", "1, Description of septic shock in notes | 2, Objective sepsis evidence | 3, Measured criteria | 4, 1 and 2 | 5, All of above")
    field("shock2_v4_card", "radio", "1, Description of cardiogenic shock in notes | 2, Objective cardiac evidence | 3, Inotropes or support | 4, 1 and 2 | 5, All of above")
    field(PHENOTYPES[-1][1], "radio", "1, Yes | 2, No")
    for name in ["ca_setting", "ca_rhythm", "bystander_cpr", "witnessed_ca", "time_to_rosc_ca", "cpr_time_ca"]:
        field(name)
    field("mortality_ca", "radio", "1, Alive | 2, Dead")
    field("sei_y_v3", "yesno")
    field("sepsis_uncertain_v3")
    return specs


def source(**values):
    return {"institution_v3": "SYNTHETIC-SITE", "record_id": "SYNTHETIC-PATIENT",
            "vid": "SYNTHETIC-VISIT", **values}


def case_for(condition, **values):
    return next(c for c in prepare_cases([source(**values)], instrument()) if c["condition"] == condition)


def proposal(status=ABSTAIN, summary="Source verification is required.", quotes=()):
    return {"phenotype_status": status, "condition_present": status in {POSITIVE, PROBABLE},
            "clinical_rationale": summary, "primary_criteria_met": [],
            "key_evidence": [{"evidence_quote": q} for q in quotes],
            "inference_backend": "hybrid:Muse-Test:required_llm"}


def test_compound_choices_are_decoded_and_originals_retained():
    case = case_for("Shock (all cause)", shock2_v4_card="5")
    obs = case["observations"][0]
    assert obs["raw_response"] == "All of above"
    assert "1. Description of cardiogenic shock" in obs["response"]
    assert "data.csv:row=2:field=shock2_v4_card" in obs["source_cells"]
    assert audit_violations(case, proposal(POSITIVE, quotes=[obs["text"]])) == []


def test_final_labels_and_outcomes_never_enter_prompts():
    row = source(sepsis_ync_v3="REFERENCE-CANARY", sei_y_v3="REFERENCE-CANARY",
                 mortality_ca="REFERENCE-CANARY", sepsis_uncertain_v3="REFERENCE-CANARY")
    for case in prepare_cases([row], instrument()):
        assert "REFERENCE-CANARY" not in json.dumps([case["record"], case["criteria"], case["observations"]])
        assert case["record"]["inference_identity_masked"] is True
        assert "SYNTHETIC-PATIENT" not in case["record"]["notes_formatted_text"]


def test_same_patient_visits_link_but_reused_visits_never_merge_patients():
    rows = [source(sep5_v3="Pneumonia", sep4_v3="1"),
            source(vid="SECOND-VISIT", shock2_v4_sep="1"),
            source(record_id="OTHER-PATIENT", shock2_v4_sep="1"),
            source(institution_v3="OTHER-SITE", shock2_v4_sep="1")]
    cases = prepare_cases(rows, instrument())
    sepsis = next(c for c in cases if c["case_id"] == "S02-P01")
    assert sepsis["linked_stay_count"] == 2
    assert sepsis["visit_id_reused"] is True
    assert {o["source_row"] for o in sepsis["observations"]} == {2, 3}
    assert any("linked stay, chronology unknown" in o["text"] for o in sepsis["observations"])
    assert audit_violations(sepsis, proposal(POSITIVE))
    other_site = next(c for c in cases if c["case_id"] == "S05-P01")
    assert other_site["linked_stay_count"] == 1
    assert other_site["visit_id_reused"] is False


def test_renal_context_supports_probable_without_forcing_upgrade():
    case = case_for("Sepsis", sep5_v3="Pneumonia", sep4_v3="1", aki1_v3="1")
    quote = case["observations"][0]["text"]
    assert not audit_violations(case, proposal(PROBABLE, quotes=[quote]))
    assert audit_violations(case, proposal(POSITIVE, quotes=[quote]))
    assert not audit_violations(case, proposal())
    no_infection = case_for("Sepsis", aki1_v3="1")
    assert audit_violations(no_infection, proposal(PROBABLE))


def test_infection_alone_and_none_of_above_stay_unknown():
    case = case_for("Sepsis", sep5_v3="Pneumonia", sep4_v3="1")
    assert audit_violations(case, proposal(PROBABLE))
    assert not audit_violations(case, proposal())
    shock = case_for("Shock (all cause)", shock1_v3="None of the above")
    assert audit_violations(shock, proposal(POSITIVE))
    assert audit_violations(shock, proposal("CONFIRMED_NEGATIVE"))


def test_numeric_sofa_without_baseline_is_only_a_probable_pathway():
    case = case_for("Sepsis", sep5_v3="Pneumonia", sep4_v3="1", sep7_v3="4")
    quote = case["observations"][0]["text"]
    assert not audit_violations(case, proposal(PROBABLE, quotes=[quote]))
    assert audit_violations(case, proposal(POSITIVE, quotes=[quote]))
    nonnumeric = case_for("Sepsis", sep5_v3="Pneumonia", sep4_v3="1", sep7_v3="No")
    assert audit_violations(nonnumeric, proposal(PROBABLE))


def test_partial_shock_criteria_differ_from_a_reported_diagnosis():
    case = case_for("Shock (all cause)", shock1_v3="systemic hypoperfusion; refractoriness")
    quote = case["observations"][0]["text"]
    assert not audit_violations(case, proposal(PROBABLE, quotes=[quote]))
    assert audit_violations(case, proposal(POSITIVE, quotes=[quote]))


def test_bnp_allows_qualified_adhf_but_not_confirmation_without_cardiac_evidence():
    case = case_for("Acute decompensated heart failure", adhf2_v3="None of the above",
                    adhf6_v3="There is no cardiologist's note.", adhf3_v3="BNP elevation")
    quote = case["observations"][0]["text"]
    assert not audit_violations(case, proposal(PROBABLE, quotes=[quote]))
    assert audit_violations(case, proposal(POSITIVE, quotes=[quote]))
    nonspecific = case_for("Acute decompensated heart failure", adhf3_v3="Troponin elevation")
    assert audit_violations(nonspecific, proposal(PROBABLE, quotes=[nonspecific["observations"][0]["text"]]))


def test_ambiguous_screens_cannot_become_measured_negatives():
    case = case_for("Delirium", del2_camicu_v3="There was no assessment or a score below threshold")
    assert audit_violations(case, proposal(summary="CAM-ICU was absent."))
    assert not audit_violations(case, proposal(summary="CAM-ICU was not assessed or below threshold."))


def test_arrest_event_is_not_vetoed_by_withheld_primary_label():
    case = case_for("Cardiac arrest", ca_setting="Out-of-hospital cardiac arrest (OHCA)",
                    witnessed_ca="Witnessed arrest", bystander_cpr="Yes", time_to_rosc_ca="20-30")
    assert audit_violations(case, proposal(summary="No explicit affirmative cardiac arrest yes/no response was supplied."))
    unknown = case_for("Cardiac arrest", ca_setting="Unknown", ca_rhythm="Unknown")
    assert not audit_violations(unknown, proposal())
    assert audit_violations(unknown, proposal(POSITIVE, quotes=[unknown["observations"][0]["text"]]))


def test_full_quotes_are_required_and_synthetic_note_metadata_is_removed():
    case = case_for("Shock (all cause)", shock2_v4_card="1")
    line = case["observations"][0]["text"]
    assert audit_violations(case, proposal(POSITIVE, quotes=["Description of cardiogenic shock"]))
    responses = [proposal(POSITIVE, quotes=[line])]
    responses[0]["key_evidence"][0].update(note_id=999, note_date="2099-01-01")
    engine = FakeEngine(responses)
    output = run_case(case, engine)
    evidence = output["verified_evidence"][0]
    assert evidence["note_id"] is None and evidence["note_date"] is None
    assert evidence["source_fields"] == ["shock2_v4_card"]


class FakeEngine:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.criteria = []

    def adjudicate_single(self, record, condition, criteria):
        self.criteria.append(criteria)
        return copy.deepcopy(next(self.responses))


def test_repairs_keep_all_feedback_and_hold_unresolved_decisions():
    case = case_for("Shock (all cause)", shock2_v4_card="1")
    line = case["observations"][0]["text"]
    engine = FakeEngine([proposal(POSITIVE, quotes=["invented quote"]),
                         proposal("CONFIRMED_NEGATIVE", quotes=[line]),
                         proposal(POSITIVE, quotes=[line])])
    output = run_case(case, engine)
    assert not output["unresolved_quality_flags"]
    assert len(output["quality_attempts"]) == 3
    assert "complete" in engine.criteria[2].lower()
    assert "exclusion" in engine.criteria[2].lower()
    held = run_case(case, FakeEngine([proposal(POSITIVE)] * 3))
    assert held["result"]["phenotype_status"] == ABSTAIN
    assert held["result"]["inference_backend"].endswith(":quality_hold")
    assert held["model_proposal_before_quality_hold"]["phenotype_status"] == POSITIVE


def test_rule_only_results_cannot_pass_the_adapter():
    case = case_for("Sepsis")
    result = {**proposal(), "inference_backend": "hybrid:fast_rule_out"}
    held = run_case(case, FakeEngine([result] * 3))
    assert held["unresolved_quality_flags"]
    assert held["result"]["inference_backend"].endswith(":quality_hold")


def test_invalid_dictionary_and_missing_linkage_fail_before_inference():
    with pytest.raises(ValueError, match="dictionary"):
        prepare_cases([source()], {})
    with pytest.raises(ValueError, match="record_id"):
        prepare_cases([{**source(), "record_id": ""}], instrument())


def test_cli_dry_run_exports_inputs_without_loading_a_model(tmp_path, monkeypatch):
    dictionary, data, out = tmp_path / "dictionary.csv", tmp_path / "data.csv", tmp_path / "run"
    specs = list(instrument().values())
    for path, rows in [(dictionary, specs), (data, [source(aki1_v3="1")])]:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    monkeypatch.delenv("ROSEGOLD_MUSE_URL", raising=False)
    assert main(["--data", str(data), "--dictionary", str(dictionary), "--output-dir", str(out), "--prepare-only"]) == 0
    assert len(json.loads((out / "model_inputs.json").read_text())) == 9
    assert json.loads((out / "run_manifest.json").read_text())["inference_run"] is False
    with pytest.raises(ValueError, match="empty"):
        main(["--data", str(data), "--dictionary", str(dictionary), "--output-dir", str(out), "--prepare-only"])


@pytest.mark.parametrize("invalid,exit_code,calls", [(False, 0, 9), (True, 2, 27)])
def test_cli_uses_actual_hybrid_path_and_records_failures(tmp_path, monkeypatch, invalid, exit_code, calls):
    import requests

    for key in ["ROSEGOLD_LLM_BACKEND", "ROSEGOLD_MUSE_URL", "ROSEGOLD_MUSE_MODEL",
                "ROSEGOLD_HYBRID_REQUIRE_LLM", "ROSEGOLD_ALLOW_MOCK", "ROSEGOLD_MUSE_TIMEOUT"]:
        monkeypatch.setenv(key, "")
    data, dictionary, out = tmp_path / "data.csv", tmp_path / "dictionary.csv", tmp_path / "run"
    for path, rows in [(dictionary, list(instrument().values())),
                       (data, [source(record_id="=SYNTHETIC-CANARY", sepsis_ync_v3="REFERENCE-CANARY")])]:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    requests_seen = []

    def post(session, url, **kwargs):
        payload = kwargs["json"]
        requests_seen.append(payload)
        assert url == "https://muse.internal/v1/chat/completions"
        assert "SYNTHETIC-CANARY" not in json.dumps(payload)
        assert "REFERENCE-CANARY" not in json.dumps(payload)
        assert payload["max_tokens"] == 8192

        class Response:
            status_code = 200

            def json(self):
                return {"id": f"synthetic-{len(requests_seen)}", "model": "wrong-model" if invalid else payload["model"],
                        "choices": [{"finish_reason": "stop", "message": {
                            "content": json.dumps(proposal()), "reasoning_content": "PRIVATE-CANARY"}}]}

        return Response()

    monkeypatch.setattr(requests.Session, "post", post)
    assert main(["--data", str(data), "--dictionary", str(dictionary), "--output-dir", str(out),
                 "--endpoint", "https://muse.internal/v1/completions", "--model", "Muse-Test"]) == exit_code
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["total_model_requests"] == len(requests_seen) == calls
    assert manifest["quality_hold_cases"] == (9 if invalid else 0)
    assert manifest["successful_required_muse_cases"] == (0 if invalid else 9)
    assert "PRIVATE-CANARY" not in (out / "model_results.jsonl").read_text()
    with (out / "review.csv").open(encoding="utf-8-sig", newline="") as handle:
        review = list(csv.DictReader(handle))
    assert len(review) == 9
    assert review[0]["Assumed patient ID"] == "'=SYNTHETIC-CANARY"
    assert all(not r[k] for r in review for k in ["Reviewer check", "Reviewer final label", "Reviewer notes"])
