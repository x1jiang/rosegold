import os
import time
import requests
import json
import pandas as pd
import subprocess

def test_full_pipeline_e2e():
    print("=== [1/6] Launching FastAPI Backend on Port 8008 for E2E Testing ===")
    env = os.environ.copy()
    env["ROSEGOLD_DATA_DIR"] = "data"
    env["ROSEGOLD_AUDIT_LOG"] = "outputs/human_audit_log.jsonl"
    
    server_process = subprocess.Popen(
        ["/Users/xiaoqianjiang/anaconda3/bin/python", "-m", "uvicorn", "app.api:app", "--port", "8008", "--host", "127.0.0.1"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    base_url = "http://127.0.0.1:8008"
    time.sleep(2) # Allow server to bind

    try:
        # Step 1: Health Check
        print("\n=== [2/6] Verifying GET /health ===")
        res = requests.get(f"{base_url}/health")
        assert res.status_code == 200, f"Health check failed: {res.text}"
        health = res.json()
        print(f"Health check OK: {json.dumps(health, indent=2)}")

        # Step 2: Visits Listing
        print("\n=== [3/6] Verifying GET /api/visits & GET /api/notes/20001 ===")
        res = requests.get(f"{base_url}/api/visits")
        assert res.status_code == 200
        visits = res.json()
        print(f"Retrieved {len(visits)} OMOP visits successfully.")
        
        res = requests.get(f"{base_url}/api/notes/20001")
        assert res.status_code == 200
        note_data = res.json()
        print(f"Retrieved Visit 20001 notes ({note_data['note_count']} notes linked).")

        # Step 3: Single Encounter Adjudications
        print("\n=== [4/6] Verifying POST /api/adjudicate/single across Phenotypes ===")
        # Test Case 1: Sepsis Case (Visit 20001)
        r1 = requests.post(f"{base_url}/api/adjudicate/single", json={
            "visit_occurrence_id": 20001,
            "target_condition": "Sepsis / Septic Shock"
        }).json()
        print(f"Visit 20001 (Sepsis Test): Status={r1['phenotype_status']}, Conf={r1['confidence_score']}, Evidence={len(r1['key_evidence'])} quotes")
        assert r1['condition_present'] is True
        assert r1['phenotype_status'] == "CONFIRMED_POSITIVE"

        # Test Case 2: Negative Control (Visit 20002)
        r2 = requests.post(f"{base_url}/api/adjudicate/single", json={
            "visit_occurrence_id": 20002,
            "target_condition": "Sepsis / Septic Shock"
        }).json()
        print(f"Visit 20002 (Control Test): Status={r2['phenotype_status']}, Conf={r2['confidence_score']}")
        assert r2['condition_present'] is False
        assert r2['phenotype_status'] == "CONFIRMED_NEGATIVE"

        # Test Case 3: Stroke Case (Visit 20003)
        r3 = requests.post(f"{base_url}/api/adjudicate/single", json={
            "visit_occurrence_id": 20003,
            "target_condition": "Acute Ischemic Stroke"
        }).json()
        print(f"Visit 20003 (Stroke Test): Status={r3['phenotype_status']}, Conf={r3['confidence_score']}")
        assert r3['condition_present'] is True
        assert r3['phenotype_status'] == "CONFIRMED_POSITIVE"

        # Step 4: Batch Adjudication
        print("\n=== [5/6] Verifying POST /api/adjudicate/batch on Full Cohort ===")
        r_batch = requests.post(f"{base_url}/api/adjudicate/batch", json={
            "target_condition": "Sepsis / Septic Shock"
        }).json()
        print(f"Batch adjudicated {len(r_batch)} encounters in one API call.")
        assert len(r_batch) == len(visits)

        # Step 5: Physician Feedback & Audit Trail
        print("\n=== [6/6] Verifying POST /api/feedback & Audit Log Persistence ===")
        fb = {
            "visit_occurrence_id": 20001,
            "person_id": 1001,
            "reviewer_id": "Dr. Gilles Clermont",
            "adjudication_status": "CONFIRMED_POSITIVE",
            "reviewer_agreement": True,
            "comments": "Agreed with LLM adjudication. Sepsis-3 criteria clearly satisfied."
        }
        res_fb = requests.post(f"{base_url}/api/feedback", json=fb)
        assert res_fb.status_code == 200
        assert os.path.exists("outputs/human_audit_log.jsonl")
        print("Audit feedback verified and logged to outputs/human_audit_log.jsonl.")

        print("\n" + "="*50)
        print(">>> ALL END-TO-END TESTS PASSED SUCCESSFULLY! <<<")
        print("="*50)

    finally:
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    test_full_pipeline_e2e()
