#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const check = process.argv[2];

const pytestCases = {
  schema: "tests/test_mimic_ext_notes.py::test_schema_accepts_official_and_rejects_missing",
  convert: "tests/test_mimic_ext_notes.py::test_notes_convert_to_omop_ids",
  load: "tests/test_mimic_ext_notes.py::test_converted_notes_load_through_chart_loader",
  labels: "tests/test_mimic_ext_notes.py::test_labels_join_dash_rules_and_gold_mismatch",
  cli: "tests/test_mimic_ext_notes.py::test_prepare_cli_writes_omop_csvs",
};

function pythonBin() {
  if (process.env.PYTHON) {
    return process.env.PYTHON;
  }
  for (const name of ["python", "python3"]) {
    const probe = spawnSync(name, ["-c", "import pytest, pandas"], { encoding: "utf8" });
    if (probe.status === 0) {
      return name;
    }
  }
  return "python3";
}

function run(command, args, options = {}) {
  const env = { ...process.env, PYTHONPATH: root, ...(options.env || {}) };
  const result = spawnSync(command, args, {
    cwd: root,
    encoding: "utf8",
    ...options,
    env,
  });
  return result;
}

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

function pass(token) {
  process.stdout.write(`${token}\n`);
}

function runPytest(nodeid) {
  const result = run(pythonBin(), ["-m", "pytest", "-p", "no:logfire", nodeid, "-q"]);
  if (result.status !== 0) {
    process.stdout.write(result.stdout || "");
    process.stderr.write(result.stderr || "");
    fail(`pytest failed for ${nodeid}`);
  }
}

if (check in pytestCases) {
  runPytest(pytestCases[check]);
  pass(`mimic-ext-notes ${check} verification passed`);
  process.exit(0);
}

if (check === "suite") {
  const result = run(pythonBin(), ["-m", "pytest", "-p", "no:logfire", "tests/", "-q"]);
  process.stdout.write(result.stdout || "");
  process.stderr.write(result.stderr || "");
  if (result.status !== 0) {
    fail("pytest suite failed");
  }
  pass("mimic-ext-notes suite verification passed");
  process.exit(0);
}

if (check === "privacy") {
  const gitignore = fs.readFileSync(path.join(root, ".gitignore"), "utf8");
  for (const line of ["data/physionet/", "data/mimic-iii-ext-notes/"]) {
    if (!gitignore.includes(line)) {
      fail(`.gitignore is missing ${line}`);
    }
  }

  const fixtureNotes = path.join(root, "tests/fixtures/mimic_iii_ext_notes/notes.csv");
  const text = fs.readFileSync(fixtureNotes, "utf8");
  const rows = text.trim().split(/\r?\n/).slice(1).filter(Boolean);
  if (rows.length === 0 || rows.length > 8) {
    fail(`committed fixture has ${rows.length} notes; expected a tiny synthetic sample`);
  }
  if (!text.includes("SYNTHETIC-FIXTURE")) {
    fail("committed fixture is missing the SYNTHETIC-FIXTURE marker");
  }

  const tracked = run("git", ["ls-files", "data/physionet", "data/mimic-iii-ext-notes"]);
  const trackedFiles = (tracked.stdout || "")
    .trim()
    .split(/\n/)
    .filter((line) => line && !line.endsWith(".gitkeep"));
  if (trackedFiles.length) {
    fail(`credentialed extract files are tracked: ${trackedFiles.join(", ")}`);
  }
  pass("mimic-ext-notes privacy verification passed");
  process.exit(0);
}

fail("usage: node scripts/verify_mimic_ext_notes.mjs <schema|convert|load|labels|cli|suite|privacy>");
