"""
1_3e_extraction_audit.py
Audit of the regex-based Context/Decision extraction (Section 4.2 of the paper).

Reads the candidate records as they were BEFORE review and re-extraction
(labeled_adrs_before_review.json), downloads each source file from GitHub, and checks
whether the extracted Context and Decision cover the whole section they were taken from.

A field counts as TRUNCATED when the extracted text is shorter than 90% of its section,
where a section runs to the next heading of the same or a higher level (headings inside
code fences are ignored). For Decision fields, a cut at a "Consequences", "Confirmation"
or "Rationale" sub-heading is NOT counted as truncation.

The script also lists the random 10% sample (seed 42) that was compared by hand against
the source files; the manual verdicts are in review_sheet_reviewed.csv (column extraction_ok).

    python 1_3e_extraction_audit.py

Output:
    extraction_audit.csv   one row per record: status and coverage ratio of both fields
    adr_sources/           cached source files (shared with 1_3d_finalize_dataset.py)
"""
import csv
import hashlib
import json
import os
import random
import re
import time
import urllib.request

IN_FILE = "labeled_adrs_before_review.json"
OUT_FILE = "extraction_audit.csv"
SRC_DIR = "adr_sources"
SEED = 42
SAMPLE_FRACTION = 0.10
MIN_COVERAGE = 0.90

HEAD = re.compile(r"^(#{1,6})\s+(.*)$")
BOLD_HEAD = re.compile(r"^\*\*[A-Z][\w /-]{2,40}\*\*:?\s*$")
ACCEPTED_DECISION_CUT = re.compile(r"consequences|confirmation|rationale", re.IGNORECASE)


def fetch_source(url):
    os.makedirs(SRC_DIR, exist_ok=True)
    path = os.path.join(SRC_DIR, hashlib.md5(url.encode("utf-8")).hexdigest()[:10] + ".md")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raw = url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/", 1)
        raw = urllib.request.quote(raw, safe=":/%")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(raw, timeout=30) as resp:
                    body = resp.read()
                with open(path, "wb") as f:
                    f.write(body)
                break
            except Exception:
                if attempt == 2:
                    return None
                time.sleep(3)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().replace("\r\n", "\n")


def analyse(src, text, is_decision):
    """Return (status, coverage ratio) for one extracted field."""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return "empty", 0.0
    pos = src.find(text)
    if pos < 0:
        pos = src.find(text[:60])
    if pos < 0:
        return "not found in source", float("nan")
    end = pos + len(text)
    before = src[:pos].rstrip("\n").split("\n")
    heading = before[-1].strip() if before else ""
    m = HEAD.match(heading)
    level = len(m.group(1)) if m else (7 if heading.startswith("**") else None)

    acc, fence = [], False
    for ln in src[pos:].split("\n"):
        s = ln.strip()
        if s.startswith("```") or s.startswith("~~~"):
            fence = not fence
        h = HEAD.match(ln) if not fence else None
        if h and level and len(h.group(1)) <= level and acc:
            break
        if level == 7 and not fence and acc and BOLD_HEAD.match(s):
            break
        acc.append(ln)
    full = "\n".join(acc).strip()
    ratio = len(text) / max(1, len(full))

    if src[:end].count("```") % 2 == 1:
        return "truncated inside code block", ratio
    if ratio < MIN_COVERAGE:
        after = src[end:].lstrip("\n").split("\n")[0] if end < len(src) else ""
        h = HEAD.match(after)
        if is_decision and h and ACCEPTED_DECISION_CUT.search(h.group(2)):
            return "ok (ends at consequences/rationale sub-heading)", ratio
        return "truncated at nested heading" if h else "truncated", ratio
    return "ok", ratio


def main():
    with open(IN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(SEED)
    n_sample = max(1, round(len(data) * SAMPLE_FRACTION))
    sample = set(rng.sample(sorted(x["url"] for x in data), n_sample))

    rows, missing = [], 0
    for x in data:
        src = fetch_source(x["url"])
        if src is None:
            missing += 1
            d_status, d_ratio, c_status, c_ratio = "source unavailable", "", "source unavailable", ""
        else:
            d_status, d_ratio = analyse(src, x["decision"], True)
            c_status, c_ratio = analyse(src, x["context"], False)
            d_ratio, c_ratio = round(d_ratio, 3), round(c_ratio, 3)
        rows.append({"url": x["url"], "repo": x["repo"], "in_manual_sample": int(x["url"] in sample),
                     "decision_status": d_status, "decision_coverage": d_ratio,
                     "context_status": c_status, "context_coverage": c_ratio,
                     "context_contains_decision": int(x["decision"].strip()[:60] in x["context"])})

    with open(OUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    d_trunc = sum(r["decision_status"].startswith("truncated") for r in rows)
    c_trunc = sum(r["context_status"].startswith("truncated") for r in rows)
    leak = sum(r["context_contains_decision"] for r in rows)
    print(f"Records checked: {n} (source unavailable: {missing})")
    print(f"Decision fields truncated: {d_trunc} ({d_trunc / n:.1%})")
    print(f"Context fields truncated : {c_trunc} ({c_trunc / n:.1%})")
    print(f"Context contains the Decision text: {leak}")
    print(f"Manual sample: {n_sample} records (in_manual_sample = 1 in {OUT_FILE}); "
          f"verdicts are in review_sheet_reviewed.csv, column extraction_ok")


if __name__ == "__main__":
    main()
