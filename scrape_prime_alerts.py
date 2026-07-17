#!/usr/bin/env python3
"""
Scrape PRIME alert pages and extract:
    object name, internal ID, RA, Dec, assessment, comment

Usage:
    python scrape_prime_alerts.py --year 2025 --out prime_2025.csv
    python scrape_prime_alerts.py --year 2025 --start 1 --end 472   # skip index scrape

Index:   https://moaprime.massey.ac.nz/alerts/index/prime/<year>
Display: https://moaprime.massey.ac.nz/alerts/display/PRIME-<year>-BLG-XXXX
"""

import argparse
import csv
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

BASE = "https://moaprime.massey.ac.nz"
INDEX_URL = BASE + "/alerts/index/prime/{year}"
DISPLAY_URL = BASE + "/alerts/display/{name}"

# Be a polite, identifiable client (you're a collaborator, but still throttle).
HEADERS = {"User-Agent": "PRIME-collab-scraper/1.0 (kde@astro.columbia.edu)"}

# Section headers that terminate the free-text Comments block.
NEXT_SECTION = re.compile(
    r"^(PSSL parameters|Photometry data file|Calibration|Event data)\s*$",
    re.IGNORECASE,
)

FIELDS = ["name", "internal_id", "ra", "dec", "assessment", "comment"]


def get_event_names(session, year):
    """Pull the list of PRIME-<year>-BLG-#### names from the index page."""
    r = session.get(INDEX_URL.format(year=year), timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    pat = re.compile(rf"PRIME-{year}-BLG-\d+")
    names = set()
    for a in soup.find_all("a", href=True):
        m = pat.search(a["href"]) or pat.search(a.get_text())
        if m:
            names.add(m.group(0))
    if not names:  # fallback: scan raw text
        names.update(pat.findall(r.text))
    # sort numerically by the trailing index
    return sorted(names, key=lambda n: int(n.rsplit("-", 1)[1]))


def parse_display(html, name):
    """Extract the requested fields from a single display page."""
    soup = BeautifulSoup(html, "html.parser")

    rec = {f: "" for f in FIELDS}
    rec["name"] = name

    # Flatten to text with line breaks; the layout is "Label:\tValue".
    lines = [ln.strip() for ln in soup.get_text("\n").splitlines()]
    lines = [ln for ln in lines if ln]

    label_map = {
        "internal id": "internal_id",
        "ra": "ra",
        "dec": "dec",
        "assessment": "assessment",
    }

    for i, ln in enumerate(lines):
        # label/value on same line ("Internal ID:\tGB111-H-1-80")
        if ":" in ln:
            label, _, val = ln.partition(":")
            key = label.strip().lower()
            if key in label_map and val.strip():
                rec[label_map[key]] = val.strip()
                continue
            # label/value split across two lines ("RA:" then value)
            if key in label_map and not val.strip() and i + 1 < len(lines):
                rec[label_map[key]] = lines[i + 1].strip()

        # Comments block: everything between "Comments" and the next section.
        if ln.strip().lower() == "comments":
            buf = []
            for nxt in lines[i + 1:]:
                if NEXT_SECTION.match(nxt):
                    break
                buf.append(nxt)
            rec["comment"] = " ".join(buf).strip()

    return rec


def fetch(session, url, retries=3, pause=1.0):
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            if attempt == retries:
                raise
            sys.stderr.write(f"  retry {attempt}/{retries} ({e})\n")
            time.sleep(pause * attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--out", default=None)
    ap.add_argument("--start", type=int, help="start index (skip index scrape)")
    ap.add_argument("--end", type=int, help="end index inclusive")
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
    ap.add_argument("--insecure", action="store_true",
                    help="skip SSL cert verification (use if cert chain won't validate)")
    args = ap.parse_args()

    out = args.out or f"prime_{args.year}.csv"
    session = requests.Session()
    session.headers.update(HEADERS)
    if args.insecure:
        session.verify = False
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if args.start and args.end:
        names = [f"PRIME-{args.year}-BLG-{i:04d}" for i in range(args.start, args.end + 1)]
    else:
        sys.stderr.write("Fetching event list from index...\n")
        names = get_event_names(session, args.year)
    sys.stderr.write(f"{len(names)} events to scrape.\n")

    rows = []
    for j, name in enumerate(names, 1):
        sys.stderr.write(f"[{j}/{len(names)}] {name}\n")
        try:
            html = fetch(session, DISPLAY_URL.format(name=name))
            rows.append(parse_display(html, name))
        except requests.RequestException as e:
            sys.stderr.write(f"  FAILED: {e}\n")
            rows.append({f: "" for f in FIELDS} | {"name": name})
        time.sleep(args.delay)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    sys.stderr.write(f"Wrote {len(rows)} rows to {out}\n")


if __name__ == "__main__":
    main()

