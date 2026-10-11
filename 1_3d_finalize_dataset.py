"""
1_3d_finalize_dataset.py
Chốt dataset sau khi rà soát. Chạy SAU 1_3b_dedupe.py + rà soát, TRƯỚC 1_4_split_dataset.py.
(Thay cho lệnh `1_3c_review.py apply`.)

Làm ba việc, theo file review_sheet_reviewed.csv:
  1. LOẠI ADR không dùng được (is_adr = n) và ADR ngoài 7 chủ đề (scope = out).
  2. SỬA NHÃN theo cột label_ok (y = giữ, số 1-7 = nhóm đúng).
  3. TRÍCH LẠI Context/Decision từ file gốc trên GitHub theo quy tắc đúng cấp heading:
     một mục kéo dài đến heading kế tiếp CÙNG CẤP HOẶC CAO HƠN (bỏ qua dòng '#' trong
     khối code). Bản cũ dừng ở mọi heading #, ##, ### nên cắt cụt các mục có heading con.
     Riêng Decision: dừng ở heading con 'Consequences' / 'Confirmation' (kiểu MADR).
     Sau đó loại ADR có Decision ngắn hơn MIN_DECISION_CHARS. ADR có Decision dài hơn
     MAX_DECISION_CHARS_FOR_TEST vẫn được giữ (dùng cho RAG pool) nhưng gắn test_eligible = false,
     để 1_4_split_dataset.py không đưa vào test set.

Cách chạy:
    python 1_3d_finalize_dataset.py

Kết quả:
    labeled_adrs.json               (GHI ĐÈ; bản trước lưu ở labeled_adrs_before_review.json)
    finalize_report.json            các con số để đưa vào bài
    finalize_detail.csv             từng ADR: giữ/loại, lý do, độ dài trước và sau khi trích lại
    adr_sources/                    file gốc đã tải (để tái lập)
"""
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

IN_FILE = "labeled_adrs.json"
SHEET = sys.argv[1] if len(sys.argv) > 1 else "review_sheet_reviewed.csv"
BACKUP = "labeled_adrs_before_review.json"
SRC_DIR = "adr_sources"
MIN_CONTEXT_CHARS = 100          # như tiêu chí lúc thu thập
MIN_DECISION_CHARS = 40
MAX_DECISION_CHARS_FOR_TEST = 4000   # ~1.000 token = giới hạn output của model ở bước 2;
                                     # reference dài hơn thì model không thể sinh lại đầy đủ

HEAD = re.compile(r"^(#{1,6})\s+(.*)$")
STOP_SUB = re.compile(r"consequences|confirmation", re.IGNORECASE)
RULE = re.compile(r"\n\s*(-{3,}|\*{3,}|_{3,})\s*$")


def read_sheet(path):
    for enc in ("utf-8-sig", "cp1258", "cp1252"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise SystemExit(f"Không đọc được {path}.")
    first = text.splitlines()[0]
    delim = ";" if first.count(";") > first.count(",") else ("\t" if "\t" in first else ",")
    return {r["url"].strip(): r for r in csv.DictReader(text.splitlines(keepends=True), delimiter=delim)}


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
            except Exception as e:
                if attempt == 2:
                    return None
                time.sleep(3)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().replace("\r\n", "\n")


def reextract(src, old_text, is_decision):
    """Trả về (văn bản mới, trạng thái). Neo tại đúng heading mà bản cũ đã khớp."""
    old = old_text.replace("\r\n", "\n").strip()
    pos = src.find(old)
    if pos < 0:
        pos = src.find(old[:60])
    if pos < 0:
        return old, "không tìm thấy trong file gốc - giữ bản cũ"
    before = src[:pos].rstrip("\n").split("\n")
    m = HEAD.match(before[-1].strip()) if before else None
    if not m:
        return old, "heading kiểu **đậm** - giữ bản cũ"
    level = len(m.group(1))
    acc, fence = [], False
    for ln in src[pos:].split("\n"):
        s = ln.strip()
        if s.startswith("```") or s.startswith("~~~"):
            fence = not fence
        h = HEAD.match(ln) if not fence else None
        if h and acc:
            if len(h.group(1)) <= level:
                break
            if is_decision and STOP_SUB.search(h.group(2)):
                break
        acc.append(ln)
    new = "\n".join(acc).strip()
    new = RULE.sub("", new).strip()
    return new, ("không đổi" if new == old else "đã trích lại")


def main():
    with open(IN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    sheet = read_sheet(SHEET)
    groups = sorted({x["group"] for x in data})
    if not os.path.exists(BACKUP):
        with open(BACKUP, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    kept, detail = [], []
    st = {"not_adr": 0, "out_of_scope": 0, "label_changed": 0, "not_in_sheet": 0,
          "decision_reextracted": 0, "context_reextracted": 0, "source_unavailable": 0,
          "decision_too_short": 0, "not_test_eligible": 0, "context_too_short": 0}
    for x in data:
        row = sheet.get(x["url"])
        d = {"url": x["url"], "repo": x["repo"], "group_before": x["group"], "group_after": "",
             "kept": 0, "reason": "", "decision_chars_before": len(x["decision"]),
             "decision_chars_after": "", "context_chars_before": len(x["context"]),
             "context_chars_after": ""}
        detail.append(d)
        if row is None:
            st["not_in_sheet"] += 1
            d["reason"] = "không có trong bảng rà soát"
            continue
        cell = lambda c: (row.get(c) or "").strip().lower()
        if cell("is_adr") != "y":
            st["not_adr"] += 1
            d["reason"] = "không phải ADR dùng được"
            continue
        if cell("scope") != "in":
            st["out_of_scope"] += 1
            d["reason"] = "ngoài 7 chủ đề"
            continue

        lab = cell("label_ok")
        if lab.isdigit() and 1 <= int(lab) <= len(groups):
            new_group = groups[int(lab) - 1]
            if new_group != x["group"]:
                st["label_changed"] += 1
                x["group_before_review"] = x["group"]
                x["group"] = new_group
        x["label_source_original"] = x.get("label_source", "unknown")
        x["label_reviewed"] = True
        d["group_after"] = x["group"]

        src = fetch_source(x["url"])
        if src is None:
            st["source_unavailable"] += 1
            notes = ["không tải được file gốc - giữ bản cũ"]
        else:
            new_ctx, s1 = reextract(src, x["context"], False)
            new_dec, s2 = reextract(src, x["decision"], True)
            st["context_reextracted"] += s1 == "đã trích lại"
            st["decision_reextracted"] += s2 == "đã trích lại"
            x["context"], x["decision"] = new_ctx, new_dec
            notes = [f"context: {s1}", f"decision: {s2}"]
        x["context_length"] = len(x["context"])
        d["decision_chars_after"], d["context_chars_after"] = len(x["decision"]), len(x["context"])

        if len(x["context"]) < MIN_CONTEXT_CHARS:
            st["context_too_short"] += 1
            d["reason"] = f"context < {MIN_CONTEXT_CHARS} ký tự"
        elif len(x["decision"]) < MIN_DECISION_CHARS:
            st["decision_too_short"] += 1
            d["reason"] = f"decision < {MIN_DECISION_CHARS} ký tự"
        else:
            x["test_eligible"] = len(x["decision"]) <= MAX_DECISION_CHARS_FOR_TEST
            if not x["test_eligible"]:
                st["not_test_eligible"] += 1
                notes.append(f"decision > {MAX_DECISION_CHARS_FOR_TEST} ký tự: chỉ dùng cho RAG pool")
            d["kept"], d["reason"] = 1, "; ".join(notes)
            kept.append(x)

    with open(IN_FILE, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    with open("finalize_detail.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
        w.writeheader()
        w.writerows(detail)

    counts = {g: sum(x["group"] == g for x in kept) for g in groups}
    eligible = {g: sum(x["group"] == g and x["test_eligible"] for x in kept) for g in groups}
    src_counts = {}
    for x in kept:
        src_counts[x["label_source_original"]] = src_counts.get(x["label_source_original"], 0) + 1
    repo_counts = {}
    for x in kept:
        repo_counts[x["repo"]] = repo_counts.get(x["repo"], 0) + 1
    top_repo = max(repo_counts.items(), key=lambda kv: kv[1])
    dl = sorted(len(x["decision"]) for x in kept)
    report = {"before": len(data), "after": len(kept), **st, "group_counts": counts,
              "group_counts_test_eligible": eligible,
              "label_source_original_counts": src_counts, "n_repos": len(repo_counts),
              "largest_repo": {"repo": top_repo[0], "n": top_repo[1]},
              "decision_chars_median": dl[len(dl) // 2], "decision_chars_max": dl[-1],
              "thresholds": {"min_context": MIN_CONTEXT_CHARS, "min_decision": MIN_DECISION_CHARS,
                             "max_decision_for_test": MAX_DECISION_CHARS_FOR_TEST}}
    with open("finalize_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Trước: {len(data)} ADR -> Sau: {len(kept)} ADR | {len(repo_counts)} repo "
          f"(lớn nhất: {top_repo[0]} = {top_repo[1]})")
    print(f"  Loại: không phải ADR {st['not_adr']} | ngoài chủ đề {st['out_of_scope']} | "
          f"decision quá ngắn {st['decision_too_short']} | context quá ngắn {st['context_too_short']}")
    print(f"  Sửa nhãn: {st['label_changed']} | trích lại decision: {st['decision_reextracted']} | "
          f"trích lại context: {st['context_reextracted']} | không tải được file gốc: {st['source_unavailable']}")
    print("  Phân bố nhóm: " + " | ".join(f"{g.split('_')[0]}={c}" for g, c in counts.items()))
    print(f"  Đủ điều kiện vào test (decision <= {MAX_DECISION_CHARS_FOR_TEST} ký tự): "
          + " | ".join(f"{g.split('_')[0]}={c}" for g, c in eligible.items())
          + f" | tổng {sum(eligible.values())}")
    print(f"  Độ dài decision: trung vị {report['decision_chars_median']} | lớn nhất {report['decision_chars_max']} ký tự")
    print("Chi tiết: finalize_detail.csv | Số liệu cho bài: finalize_report.json")


if __name__ == "__main__":
    main()
