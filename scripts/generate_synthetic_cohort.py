import os
import pandas as pd
import datetime

def generate_synthetic_omop_cohort(n_patients=20):
    visits = []
    notes = []

    cases = [
        {
            "id": 20001, "pid": 1001, "condition": "Septic Shock (Klebsiella urosepsis)",
            "start": "2026-03-01", "end": "2026-03-08",
            "notes": [
                ("ED Note", "74yo F nursing home resident with fever 39.1C, HR 128, BP 76/42 (MAP 53), lactate 4.2. Urosepsis. Starting Norepinephrine. Transfer to MICU."),
                ("MICU Day 2", "Intubated on Levophed 0.14 mcg/kg/min. Blood culture positive for ESBL Klebsiella. Lactate clearing 2.4. AKI with Cr 2.8."),
                ("Discharge Summary", "Severe Sepsis and Septic Shock secondary to Klebsiella urosepsis (Resolved). Completed 7 days IV Cefepime.")
            ]
        },
        {
            "id": 20002, "pid": 1002, "condition": "Elective Knee Arthroplasty (Negative Control)",
            "start": "2026-03-05", "end": "2026-03-07",
            "notes": [
                ("Pre-Op H&P", "62yo M admitted for elective right total knee arthroplasty. Normal vitals. WBC 6.1, Cr 0.8."),
                ("Discharge Summary", "Uncomplicated right TKR. Afebrile throughout stay. Incision clean. Discharged home in stable condition.")
            ]
        },
        {
            "id": 20003, "pid": 1003, "condition": "Acute Ischemic Stroke (LVO M1 MCA)",
            "start": "2026-03-10", "end": "2026-03-16",
            "notes": [
                ("Neurology Consult", "68yo M with acute right hemiplegia and aphasia, NIHSS 18. CTA shows Left M1 MCA occlusion. Administered TNK, emergent thrombectomy TICI 3."),
                ("Discharge Summary", "Acute Ischemic Stroke Left MCA territory. NIHSS improved to 3. Started Apixaban for new atrial fibrillation.")
            ]
        },
        {
            "id": 20004, "pid": 1004, "condition": "Community-Acquired Pneumonia (Mild Floor)",
            "start": "2026-03-12", "end": "2026-03-15",
            "notes": [
                ("Admission H&P", "55yo M with productive cough, T 38.3C, HR 98, BP 118/74, CXR shows right lower lobe consolidation. Lactate 1.2. Ceftriaxone initiated."),
                ("Discharge Summary", "Uncomplicated CAP. Afebrile 48 hours. No sepsis or organ dysfunction. Discharged home on oral Levofloxacin.")
            ]
        },
        {
            "id": 20005, "pid": 1005, "condition": "Trauma MVA (Non-infectious SIRS mimic)",
            "start": "2026-03-18", "end": "2026-03-22",
            "notes": [
                ("Trauma H&P", "29yo M post-MVA with rib fractures. HR 114, WBC 14.8 due to pain/stress. No infection."),
                ("Discharge Summary", "Blunt chest trauma. Pain controlled, afebrile, normal labs. Discharged home.")
            ]
        },
        {
            "id": 20006, "pid": 1006, "condition": "Severe ARDS secondary to Viral Pneumonia",
            "start": "2026-03-20", "end": "2026-03-29",
            "notes": [
                ("ED Note", "45yo F with acute respiratory distress, SpO2 82% on RA, bilateral diffuse ground glass opacities. COVID-19 PCR positive."),
                ("ICU Progress Note", "Severe ARDS with PaO2/FiO2 ratio 88 mmHg. Paralyzed and placed in prone position for 16 hours. Inhaled Epoprostenol started."),
                ("Discharge Summary", "Acute Respiratory Distress Syndrome (Severe ARDS) successfully weaned from mechanical ventilation. Discharged to rehab.")
            ]
        },
        {
            "id": 20007, "pid": 1007, "condition": "Acute Kidney Injury Stage 3 (Rhabdomyolysis)",
            "start": "2026-03-22", "end": "2026-03-28",
            "notes": [
                ("ED H&P", "80yo M found down for 18 hours. Serum CK 45,000 U/L, Creatinine 4.6 mg/dL (baseline 1.0), dark tea-colored urine."),
                ("Nephrology Consult", "Severe pigment nephropathy AKI KDIGO Stage 3. Aggressive isotonic bicarbonate infusion."),
                ("Discharge Summary", "Acute Kidney Injury secondary to rhabdomyolysis, renal function recovered without need for hemodialysis (Cr 1.4).")
            ]
        },
        {
            "id": 20008, "pid": 1008, "condition": "Acute Appendicitis with Localized Peritonitis",
            "start": "2026-03-25", "end": "2026-03-27",
            "notes": [
                ("Surgery Consult", "24yo M with RLQ pain, McBurney tenderness, T 37.9C, WBC 15.2. CT abdomen confirms acute suppurative appendicitis."),
                ("Operative Report", "Laparoscopic appendectomy performed. Appendix non-perforated."),
                ("Discharge Summary", "Acute simple appendicitis s/p lap appendectomy. Discharged home tolerating regular diet.")
            ]
        },
        {
            "id": 20009, "pid": 1009, "condition": "Bacteremic Sepsis (Staph aureus cellulitis)",
            "start": "2026-03-26", "end": "2026-04-02",
            "notes": [
                ("Admission Note", "58yo diabetic M with expanding lower extremity erythema, T 39.4C, HR 122, BP 88/54, WBC 21.0. Blood cultures positive for MSSA."),
                ("ID Consult", "MSSA bacteremia secondary to severe cellulitis. Cefazolin 2g IV q8h. Echocardiogram negative for endocarditis."),
                ("Discharge Summary", "MSSA Sepsis with bacteremia and lower extremity cellulitis. Clinical cure.")
            ]
        },
        {
            "id": 20010, "pid": 1010, "condition": "Aneurysmal Subarachnoid Hemorrhage (SAH)",
            "start": "2026-03-28", "end": "2026-04-06",
            "notes": [
                ("Neurocritical Care H&P", "51yo F with sudden 'worst headache of life' (thunderclap), Hunt-Hess 3, Fisher Grade 3 SAH from anterior communicating aneurysm."),
                ("Endovascular Note", "Successful coil embolization of ACoA aneurysm."),
                ("Discharge Summary", "Aneurysmal SAH status post endovascular coiling. Transferred to neuro-rehab.")
            ]
        },
        {
            "id": 20011, "pid": 1011, "condition": "Acute Decompensated Heart Failure (HFrEF)",
            "start": "2026-04-01", "end": "2026-04-05",
            "notes": [
                ("Cardiology H&P", "71yo M with 15lb weight gain, orthopnea, bilateral 3+ pitting edema, BNP 3,400 pg/mL, afebrile."),
                ("Discharge Summary", "Acute exacerbation of chronic systolic heart failure (EF 25%). Successfully diuresed with IV Furosemide (total net negative 6L).")
            ]
        },
        {
            "id": 20012, "pid": 1012, "condition": "Post-operative Sepsis (Anastomotic Leak)",
            "start": "2026-04-03", "end": "2026-04-12",
            "notes": [
                ("Surgery Note POD 4", "65yo F s/p sigmoid resection develops new tachycardia HR 135, T 39.2C, BP 82/50, rigid abdomen. CT shows anastomotic dehiscence and free air."),
                ("ICU Note", "Post-op septic shock s/p emergent exploratory laparotomy and Hartmann procedure. On Norepinephrine."),
                ("Discharge Summary", "Post-operative intra-abdominal sepsis and septic shock resolved.")
            ]
        },
        {
            "id": 20013, "pid": 1013, "condition": "Elective Cataract Extraction (Negative Control)",
            "start": "2026-04-05", "end": "2026-04-05",
            "notes": [
                ("Ophthalmology H&P", "72yo F elective phacoemulsification with intraocular lens placement. Uncomplicated outpatient procedure.")
            ]
        },
        {
            "id": 20014, "pid": 1014, "condition": "Transient Ischemic Attack (TIA)",
            "start": "2026-04-07", "end": "2026-04-09",
            "notes": [
                ("Neurology Consult", "60yo M with 20-minute episode of left arm weakness and dysarthria, completely resolved on arrival. MRI brain DWI negative for acute infarct."),
                ("Discharge Summary", "Transient Ischemic Attack (TIA), ABCD2 score 4. Started on Dual Antiplatelet Therapy (DAPT).")
            ]
        },
        {
            "id": 20015, "pid": 1015, "condition": "Severe Acute Pancreatitis with Necrosis",
            "start": "2026-04-10", "end": "2026-04-18",
            "notes": [
                ("Admission H&P", "42yo M with severe epigastric pain radiating to back, Lipase 4,800 U/L, SIRS positive, hemoconcentration with Hct 49%."),
                ("GI Progress Note", "Severe gallstone pancreatitis with 30% pancreatic parenchymal necrosis. ICU monitoring for multi-organ failure."),
                ("Discharge Summary", "Severe Acute Pancreatitis (Balthazar E). Organ failure resolved with aggressive hydration.")
            ]
        },
        {
            "id": 20016, "pid": 1016, "condition": "Elective Inguinal Hernia Repair (Control)",
            "start": "2026-04-12", "end": "2026-04-13",
            "notes": [
                ("Surgical H&P", "48yo M elective robotic inguinal hernia repair. Uncomplicated recovery, discharged next morning.")
            ]
        },
        {
            "id": 20017, "pid": 1017, "condition": "Pyelonephritis without Sepsis (Floor Admission)",
            "start": "2026-04-14", "end": "2026-04-16",
            "notes": [
                ("Medicine H&P", "31yo F with right flank pain, CVA tenderness, T 38.5C, HR 92, BP 112/70. Urine culture positive for E. coli. Lactate normal at 1.1."),
                ("Discharge Summary", "Acute uncomplicated pyelonephritis. Responded to IV Ceftriaxone. No sepsis or hemodynamic instability.")
            ]
        },
        {
            "id": 20018, "pid": 1018, "condition": "Cardiogenic Shock (Acute STEMI)",
            "start": "2026-04-18", "end": "2026-04-24",
            "notes": [
                ("Cath Lab Note", "64yo M anterior STEMI with 100% LAD occlusion. Primary PCI with DES. Hypotension and pulmonary edema. Intra-aortic balloon pump placed."),
                ("CCU Progress Note", "Cardiogenic shock secondary to acute myocardial infarction. Dobutamine infusion. Afebrile, WBC normal, no infection."),
                ("Discharge Summary", "Acute STEMI complicated by cardiogenic shock, stabilized and weaned off inotropic support.")
            ]
        },
        {
            "id": 20019, "pid": 1019, "condition": "Neutropenic Fever / Sepsis (Oncology)",
            "start": "2026-04-20", "end": "2026-04-27",
            "notes": [
                ("Oncology Admission", "53yo F post-chemotherapy with Absolute Neutrophil Count (ANC) 120 /mcL, T 39.5C, HR 125, BP 84/48. Urgent Cefepime + Vancomycin."),
                ("ICU Progress Note", "Neutropenic septic shock. G-CSF support. Blood cultures grew Pseudomonas aeruginosa."),
                ("Discharge Summary", "Pseudomonas neutropenic septic shock resolved with neutrophil recovery.")
            ]
        },
        {
            "id": 20020, "pid": 1020, "condition": "Diabetic Ketoacidosis (DKA) without Sepsis",
            "start": "2026-04-25", "end": "2026-04-28",
            "notes": [
                ("ED H&P", "22yo M with Type 1 Diabetes, blood glucose 640 mg/dL, pH 7.12, anion gap 26, beta-hydroxybutyrate > 6.0. Afebrile, chest x-ray clear."),
                ("ICU Note", "DKA protocol with IV regular insulin and potassium replacement. Anion gap closed within 14 hours."),
                ("Discharge Summary", "Resolved Diabetic Ketoacidosis. Transitioned back to subcutaneous insulin glargine/lispro.")
            ]
        }
    ]

    note_id_counter = 40001
    for c in cases:
        visits.append({
            "visit_occurrence_id": c["id"],
            "person_id": c["pid"],
            "visit_concept_id": 9201,
            "visit_start_date": c["start"],
            "visit_end_date": c["end"],
            "visit_type_concept_id": 44818518,
            "care_site_id": 101
        })
        for title, text in c["notes"]:
            notes.append({
                "note_id": note_id_counter,
                "person_id": c["pid"],
                "visit_occurrence_id": c["id"],
                "note_date": c["start"],
                "note_datetime": f"{c['start']} 10:00:00",
                "note_type_concept_id": 44814637,
                "note_title": title,
                "note_text": text
            })
            note_id_counter += 1

    df_v = pd.DataFrame(visits)
    df_n = pd.DataFrame(notes)

    os.makedirs("data", exist_ok=True)
    df_v.to_csv("data/synthetic_visits.csv", index=False)
    df_n.to_csv("data/synthetic_notes.csv", index=False)
    print(f"Generated {len(df_v)} synthetic visits and {len(df_n)} clinical notes.")

if __name__ == "__main__":
    generate_synthetic_omop_cohort(20)
