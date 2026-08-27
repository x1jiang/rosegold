import pytest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["engine_ready"] is True
    assert "model_name" in data

def test_get_visits_endpoint():
    response = client.get("/api/visits")
    assert response.status_code == 200
    visits = response.json()
    assert len(visits) >= 5
    first_v = visits[0]
    assert "visit_occurrence_id" in first_v
    assert "person_id" in first_v
    assert "note_count" in first_v

def test_get_visit_notes_endpoint():
    response = client.get("/api/notes/20001")
    assert response.status_code == 200
    data = response.json()
    assert data["visit_occurrence_id"] == 20001
    assert "notes_formatted_text" in data
    assert "Septic Shock" in data["notes_formatted_text"] or "MICU" in data["notes_formatted_text"]

def test_adjudicate_single_visit():
    payload = {
        "visit_occurrence_id": 20001,
        "target_condition": "Sepsis / Septic Shock"
    }
    response = client.post("/api/adjudicate/single", json=payload)
    assert response.status_code == 200
    adj = response.json()
    assert adj["visit_occurrence_id"] == 20001
    assert adj["condition_present"] is True
    assert adj["phenotype_status"] == "CONFIRMED_POSITIVE"
    assert adj["confidence_score"] > 0.8
    assert len(adj["key_evidence"]) > 0

def test_adjudicate_batch_visits():
    payload = {
        "visit_occurrence_ids": [20001, 20002, 20003],
        "target_condition": "Sepsis / Septic Shock"
    }
    response = client.post("/api/adjudicate/batch", json=payload)
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 3
    assert results[0]["condition_present"] is True # 20001 Sepsis
    assert results[1]["condition_present"] is False # 20002 Knee surgery

def test_record_physician_feedback():
    feedback = {
        "visit_occurrence_id": 20001,
        "person_id": 1001,
        "reviewer_id": "md_expert_01",
        "adjudication_status": "CONFIRMED_POSITIVE",
        "reviewer_agreement": True,
        "comments": "Agreed. Clear documentation of septic shock secondary to Klebsiella urosepsis."
    }
    response = client.post("/api/feedback", json=feedback)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
