# Gates: MIMIC-III-Ext-Notes readiness

OWNS: GATES.md, app/mimic_ext_notes.py, app/omop_loader.py, app/adjudicator.py, app/ui.py, app/api.py, scripts/prepare_mimic_iii_ext_notes.py, scripts/eval_mimic_iii_ext_notes.py, scripts/verify_mimic_ext_notes.mjs, tests/conftest.py, tests/test_mimic_ext_notes.py, tests/fixtures/mimic_iii_ext_notes/**, .gitignore, README.md

Scope: Rose Gold can validate, convert, load, and score MIMIC-III-Ext-Notes v1.0.0 without committing credentialed files.

- [x] G0: this ledger states outcomes that can fail
  CHECK: node /Users/xiaoqianjiang/.claude/skills/unlazy/scripts/gate-lint.mjs GATES.md
  EXPECT: LINT OK
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/xiaoqianjiang/Dropbox/cursor_projects/mac/rosegold; path=6856f45d0e16/39 entries; EXPECT=matched; output-sha256=48630b7361dd44ee870917b12c3d19b9d7bdea738aaca16bb04d4cab83b772d2; output-bytes=8

- [x] G1: official notes and labels columns are accepted and a missing required column is rejected
  CHECK: node scripts/verify_mimic_ext_notes.mjs schema
  EXPECT: mimic-ext-notes schema verification passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/xiaoqianjiang/Dropbox/cursor_projects/mac/rosegold; path=6856f45d0e16/39 entries; EXPECT=matched; output-sha256=c1aeeb162cb333194abcd1572ab9b5fac576e681e9414b7c9fd57538ce10f247; output-bytes=43

- [x] G2: fixture notes map row_id hadm_id subject_id text onto OMOP note_id visit_occurrence_id person_id note_text
  CHECK: node scripts/verify_mimic_ext_notes.mjs convert
  EXPECT: mimic-ext-notes convert verification passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/xiaoqianjiang/Dropbox/cursor_projects/mac/rosegold; path=6856f45d0e16/39 entries; EXPECT=matched; output-sha256=3eb5abc6bc4ee01730443d2e4d3bf472395e8f6c4dd4c83c8261d88fcd33b318; output-bytes=44

- [x] G3: converted MIMIC notes load through the chart loader and include the nursing note body
  CHECK: node scripts/verify_mimic_ext_notes.mjs load
  EXPECT: mimic-ext-notes load verification passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/xiaoqianjiang/Dropbox/cursor_projects/mac/rosegold; path=6856f45d0e16/39 entries; EXPECT=matched; output-sha256=016b972fa70713471573a0b907506e5e7bdc307a5a66d84bfba44cafb104915a; output-bytes=41

- [x] G4: labels join on row_id, enforce detection-no dash rules, and expose a phenotype gold mismatch
  CHECK: node scripts/verify_mimic_ext_notes.mjs labels
  EXPECT: mimic-ext-notes labels verification passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/xiaoqianjiang/Dropbox/cursor_projects/mac/rosegold; path=6856f45d0e16/39 entries; EXPECT=matched; output-sha256=11e2d9b1d01beb5fcbc01b68a1a28fcfc894d738eb92e1bba61b911715cd2221; output-bytes=43

- [x] G5: prepare CLI writes OMOP CSVs from a notes.csv plus labels.csv directory
  CHECK: node scripts/verify_mimic_ext_notes.mjs cli
  EXPECT: mimic-ext-notes cli verification passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/xiaoqianjiang/Dropbox/cursor_projects/mac/rosegold; path=6856f45d0e16/39 entries; EXPECT=matched; output-sha256=b51d8762089ea97dc3d21debb6fe71bfd8bff5ff0a3a7abb4ae0d25ae7fc651d; output-bytes=40

- [x] G6: the repository pytest suite exits zero after the MIMIC adapter lands
  CHECK: node scripts/verify_mimic_ext_notes.mjs suite
  EXPECT: mimic-ext-notes suite verification passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/xiaoqianjiang/Dropbox/cursor_projects/mac/rosegold; path=6856f45d0e16/39 entries; EXPECT=matched; output-sha256=5452f888a2103bb0aaacffe4d97cf0f1da001f2926758b639bb5b75f49596583; output-bytes=4531

- [x] G7: credentialed extract directories are gitignored and the committed fixture stays a tiny synthetic sample
  CHECK: node scripts/verify_mimic_ext_notes.mjs privacy
  EXPECT: mimic-ext-notes privacy verification passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/Users/xiaoqianjiang/Dropbox/cursor_projects/mac/rosegold; path=6856f45d0e16/39 entries; EXPECT=matched; output-sha256=51b70637d61df3c0e5ace2300bc3e6a225cb367adc98b41881c3a4830b15e701; output-bytes=44
