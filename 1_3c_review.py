"""
1_3c_review.py
Rà soát bằng mắt TRƯỚC khi tách dataset. Chạy sau 1_3b_dedupe.py, trước 1_4_split_dataset.py.
Gộp ba việc vào một file Excel/CSV:
  - is_adr        : file có phải ADR thật không (loại README, SKILL.md, bài giải thích code...)
  - label_ok      : nhãn nhóm G1-G7 có đúng không                           [R3#14]
  - extraction_ok : regex cắt Context/Decision có đúng không (mẫu 10%)      [R3#13]

BƯỚC 1 - xuất file để duyệt:
    python 1_3c_review.py export
    -> review_sheet.csv (mở bằng Excel)

BƯỚC 2 - điền vào 4 cột cuối rồi LƯU LẠI dạng CSV:
    is_adr        : y / n
    label_ok      : y nếu nhãn đúng, hoặc gõ số 1-7 = nhóm đúng
    extraction_ok : y / n   (chỉ cần điền ở dòng có in_audit_sample = 1; mở url để đối chiếu)
    note          : ghi chú tùy ý
  Dòng có must_review = 1 là BẮT BUỘC duyệt. Dòng khác để trống = giữ nguyên.

BƯỚC 3 - áp dụng kết quả duyệt vào labeled_adrs.json:
    python 1_3c_review.py apply
    -> loại dòng is_adr = n hoặc extraction_ok = n, sửa nhãn, in các con số để đưa vào bài
    -> bản gốc lưu ở labeled_adrs_before_review.json

TÙY CHỌN - hai người duyệt độc lập (mỗi người một bản copy của review_sheet.csv):
    python 1_3c_review.py agree nguoi_A.csv nguoi_B.csv
    -> Cohen's kappa cho nhãn nhóm. Sau đó thống nhất thành một file rồi mới chạy apply.
"""
import csv
import json
import os
import random
import sys

IN_FILE = "labeled_adrs.json"
SHEET = "review_sheet.csv"
BACKUP = "labeled_adrs_before_review.json"
REPORT = "review_report.json"
SEED = 42
AUDIT_FRACTION = 0.10
PREVIEW = 400

ADR_PATH_HINTS = ["adr", "decision"]
BAD_NAME_HINTS = ["readme", "skill.md", "template", "changelog", "index.md"]

COLUMNS = ["must_review", "path_flag", "in_audit_sample", "label_source", "group", "repo",
           "path", "url", "decision_chars", "context_preview", "decision_preview",
           "is_adr", "label_ok", "extraction_ok", "note"]


def path_flag(item):
    p = (item.get("path") or item.get("file") or "").lower()
    name = p.split("/")[-1]
    if any(h in name for h in BAD_NAME_HINTS):
        return "tên file không giống ADR"
    if not any(h in p for h in ADR_PATH_HINTS):
        return "đường dẫn không có adr/decision"
    return ""


def load_data():
    with open(IN_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def read_sheet(path):
    """Đọc CSV đã qua Excel: chịu được BOM, dấu ; và bảng mã Windows."""
    for enc in ("utf-8-sig", "cp1258", "cp1252"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise SystemExit(f"Không đọc được {path}. Lưu lại bằng 'CSV UTF-8' trong Excel.")
    first = text.splitlines()[0]
    delim = ";" if first.count(";") > first.count(",") else ("\t" if "\t" in first else ",")
    rows = list(csv.DictReader(text.splitlines(keepends=True), delimiter=delim))
    if not rows or "url" not in rows[0]:
        raise SystemExit(f"{path} thiếu cột url - đừng xóa hay đổi tên cột khi duyệt.")
    return {r["url"].strip(): r for r in rows}


def cell(row, col):
    return (row.get(col) or "").strip().lower()


def final_group(row, groups):
    """Nhãn sau khi duyệt; None nếu người duyệt chưa điền hoặc điền không hợp lệ."""
    v = cell(row, "label_ok")
    if v in ("y", "yes"):
        return row["group"].strip()
    if v.isdigit() and 1 <= int(v) <= len(groups):
        return groups[int(v) - 1]
    return None


def cmd_export():
    data = load_data()
    rng = random.Random(SEED)
    n_audit = max(1, round(len(data) * AUDIT_FRACTION))
    audit = set(rng.sample(sorted(x["url"] for x in data), n_audit))

    rows = []
    for x in data:
        flag = path_flag(x)
        in_audit = x["url"] in audit
        src = x.get("label_source", "")
        must = bool(flag) or in_audit or src == "search"
        rows.append({
            "must_review": int(must), "path_flag": flag, "in_audit_sample": int(in_audit),
            "label_source": src, "group": x["group"], "repo": x["repo"],
            "path": x.get("path", ""), "url": x["url"],
            "decision_chars": len(x.get("decision") or ""),
            "context_preview": (x.get("context") or "")[:PREVIEW].replace("\n", " "),
            "decision_preview": (x.get("decision") or "")[:PREVIEW].replace("\n", " "),
            "is_adr": "", "label_ok": "", "extraction_ok": "", "note": "",
        })
    rows.sort(key=lambda r: (-r["must_review"], r["path_flag"] == "", r["group"], r["url"]))

    with open(SHEET, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    groups = sorted({x["group"] for x in data})
    print(f"Đã xuất {len(rows)} ADR ra {SHEET}")
    print(f"  bắt buộc duyệt (must_review = 1): {sum(r['must_review'] for r in rows)}")
    print(f"  - bị gắn cờ đường dẫn : {sum(bool(r['path_flag']) for r in rows)}")
    print(f"  - nhãn từ batch 5     : {sum(r['label_source'] == 'search' for r in rows)}")
    print(f"  - mẫu audit trích xuất: {n_audit} (điền extraction_ok cho các dòng này)")
    print("Số dùng cho cột label_ok: " + " | ".join(f"{i}={g}" for i, g in enumerate(groups, 1)))


def cmd_apply(sheet_path):
    data = load_data()
    sheet = read_sheet(sheet_path)
    groups = sorted({x["group"] for x in data})
    if not os.path.exists(BACKUP):
        with open(BACKUP, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    kept = []
    st = {"is_adr_checked": 0, "not_adr": 0, "label_checked": 0, "label_changed": 0,
          "extraction_audited": 0, "extraction_bad": 0, "must_review_unfilled": 0}
    for x in data:
        row = sheet.get(x["url"])
        if row is None:
            kept.append(x)
            continue
        is_adr, extr = cell(row, "is_adr"), cell(row, "extraction_ok")
        if cell(row, "must_review") == "1" and not is_adr:
            st["must_review_unfilled"] += 1
        if is_adr:
            st["is_adr_checked"] += 1
        if is_adr in ("n", "no"):
            st["not_adr"] += 1
            continue
        if extr:
            st["extraction_audited"] += 1
        if extr in ("n", "no"):
            st["extraction_bad"] += 1
            continue
        new_group = final_group(row, groups)
        if new_group is not None:
            st["label_checked"] += 1
            if new_group != x["group"]:
                st["label_changed"] += 1
                x["group_before_review"] = x["group"]
                x["group"] = new_group
            x["label_reviewed"] = True
        kept.append(x)

    with open(IN_FILE, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    src = {}
    for x in kept:
        key = x.get("label_source", "unknown")
        src[key] = src.get(key, 0) + 1
    counts = {g: sum(x["group"] == g for x in kept) for g in groups}
    report = {"before": len(data), "after": len(kept), **st,
              "label_source_counts": src, "group_counts": counts,
              "n_repos": len({x["repo"] for x in kept})}
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Trước: {len(data)} ADR -> Sau: {len(kept)} ADR | {report['n_repos']} repo")
    print(f"  Kiểm tra 'có phải ADR': {st['is_adr_checked']} dòng, loại {st['not_adr']}")
    print(f"  Audit trích xuất      : {st['extraction_audited']} dòng, sai {st['extraction_bad']} (đã loại)")
    print(f"  Kiểm tra nhãn         : {st['label_checked']} dòng, sửa {st['label_changed']}")
    print(f"  Nguồn nhãn còn lại    : {src}")
    print("  Phân bố nhóm          : " + " | ".join(f"{g.split('_')[0]}={c}" for g, c in counts.items()))
    if st["must_review_unfilled"]:
        print(f"[CẢNH BÁO] {st['must_review_unfilled']} dòng must_review = 1 CHƯA được duyệt.")
    print(f"Các con số này đã lưu ở {REPORT} để đưa vào mục 4.2 và Threats to Validity.")


def cmd_agree(path_a, path_b):
    from sklearn.metrics import cohen_kappa_score
    a, b = read_sheet(path_a), read_sheet(path_b)
    groups = sorted({r["group"].strip() for r in a.values()})
    la, lb, adr_a, adr_b = [], [], [], []
    for url in a.keys() & b.keys():
        ga, gb = final_group(a[url], groups), final_group(b[url], groups)
        if ga and gb:
            la.append(ga)
            lb.append(gb)
        ia, ib = cell(a[url], "is_adr"), cell(b[url], "is_adr")
        if ia and ib:
            adr_a.append(ia[0])
            adr_b.append(ib[0])
    if len(la) < 2:
        raise SystemExit("Hai file chưa có đủ dòng cùng được điền label_ok.")
    agree = sum(x == y for x, y in zip(la, lb)) / len(la)
    print(f"Nhãn nhóm: {len(la)} ADR cả hai cùng duyệt | đồng thuận thô {agree:.1%} | "
          f"Cohen's kappa = {cohen_kappa_score(la, lb):.3f}")
    if len(adr_a) >= 2 and len(set(adr_a + adr_b)) > 1:
        print(f"Có phải ADR: {len(adr_a)} dòng | Cohen's kappa = {cohen_kappa_score(adr_a, adr_b):.3f}")
    diff = [u for u in a.keys() & b.keys()
            if final_group(a[u], groups) and final_group(b[u], groups)
            and final_group(a[u], groups) != final_group(b[u], groups)]
    for u in diff:
        print(f"  KHÁC NHAU: {u}  A={final_group(a[u], groups)}  B={final_group(b[u], groups)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["export"]:
        cmd_export()
    elif args[:1] == ["apply"]:
        cmd_apply(args[1] if len(args) > 1 else SHEET)
    elif args[:1] == ["agree"] and len(args) == 3:
        cmd_agree(args[1], args[2])
    else:
        print(__doc__)
