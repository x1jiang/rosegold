#!/usr/bin/env python3
"""Drive the live Streamlit UI: adjudicate, sign off, save criteria."""

import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "https://rosegold-eoohrvwf7q-uc.a.run.app/"


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel="chrome")
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.goto(URL, wait_until="domcontentloaded", timeout=90000)
        page.get_by_text("Persistent GCS").wait_for(timeout=60000)
        print("sidebar_gcs_ok")

        page.get_by_role("button", name="Adjudicate this Encounter").click()
        page.get_by_role("button", name="Sign & Save Verification").wait_for(timeout=60000)
        print("adjudicate_ok")

        comments = page.get_by_label("Clinical Review Comments")
        comments.fill("GCS persist check: visit 20001 agree with LLM")
        page.get_by_role("button", name="Sign & Save Verification").click()
        page.get_by_text("Verification saved").wait_for(timeout=30000)
        print("signoff_ok")

        page.get_by_role("tab", name="Phenotype Rules").click()
        page.get_by_role("button", name="Save Updated Phenotype Criteria").click()
        page.get_by_text("Criteria saved").wait_for(timeout=30000)
        print("criteria_ok")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
