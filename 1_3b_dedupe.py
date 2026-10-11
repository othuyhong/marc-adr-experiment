"""
1_3b_dedupe.py
Loại ADR trùng lặp khỏi labeled_adrs.json. Chạy SAU khi gán nhãn xong và TRƯỚC
khi chạy 1_4_split_dataset.py.

Vì sao cần: script tách dữ liệu chỉ kiểm tra rò rỉ theo URL. Nếu cùng một ADR
xuất hiện ở 2 URL khác nhau (bản sao trong nhiều thư mục, bản dịch giữ nguyên
thuật ngữ...), một bản có thể rơi vào test set và bản kia vào RAG pool -> RAG
"nhìn thấy đáp án" mà kiểm tra URL không phát hiện được.

Cách chạy (cùng thư mục với labeled_adrs.json):
    pip install scikit-learn
    python 1_3b_dedupe.py

Hai lớp lọc:
  1. Context giống hệt nhau (sau khi chuẩn hóa khoảng trắng/chữ hoa) -> giữ bản đầu
  2. Context+Decision gần giống nhau (TF-IDF cosine >= SIM_THRESHOLD) -> giữ bản đầu

Kết quả:
  - labeled_adrs.json               : đã loại trùng (GHI ĐÈ)
  - labeled_adrs_before_dedupe.json : bản gốc trước khi loại (chỉ tạo lần chạy đầu)
  - dedupe_report.csv               : ADR nào bị loại, trùng với ADR nào, vì sao
"""
import csv
import json
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

IN_FILE = "labeled_adrs.json"
BACKUP_FILE = "labeled_adrs_before_dedupe.json"
REPORT_FILE = "dedupe_report.csv"
SIM_THRESHOLD = 0.90


def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def main():
    with open(IN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not os.path.exists(BACKUP_FILE):
        with open(BACKUP_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    n0 = len(data)
    removed = []          # (index_bị_loại, index_được_giữ, lý_do, sim)
    gone = set()

    # Lớp 1: context giống hệt
    first_by_ctx = {}
    for i, x in enumerate(data):
        key = norm(x.get("context"))
        if key in first_by_ctx:
            removed.append((i, first_by_ctx[key], "context giống hệt", 1.0))
            gone.add(i)
        else:
            first_by_ctx[key] = i

    # Lớp 2: gần giống (TF-IDF trên Context + Decision)
    keep_idx = [i for i in range(n0) if i not in gone]
    if len(keep_idx) > 1:
        texts = [norm(data[i].get("context")) + " " + norm(data[i].get("decision"))
                 for i in keep_idx]
        try:
            mat = TfidfVectorizer(stop_words="english").fit_transform(texts)
            sims = cosine_similarity(mat)
            for a in range(len(keep_idx)):
                if keep_idx[a] in gone:
                    continue
                for b in range(a + 1, len(keep_idx)):
                    if keep_idx[b] in gone:
                        continue
                    if sims[a][b] >= SIM_THRESHOLD:
                        removed.append((keep_idx[b], keep_idx[a],
                                        f"gần giống (cosine>={SIM_THRESHOLD})",
                                        round(float(sims[a][b]), 3)))
                        gone.add(keep_idx[b])
        except ValueError:
            print("Không tính được TF-IDF (văn bản rỗng?) - bỏ qua lớp lọc 2.")

    clean = [x for i, x in enumerate(data) if i not in gone]

    with open(IN_FILE, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

    with open(REPORT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["lý_do", "similarity", "group", "bị_loại_repo", "bị_loại_file",
                    "giữ_lại_repo", "giữ_lại_file"])
        for i, k, why, s in removed:
            w.writerow([why, s, data[i].get("group", ""), data[i]["repo"], data[i]["file"],
                        data[k]["repo"], data[k]["file"]])

    print(f"Trước: {n0} ADR  ->  Sau khi loại trùng: {len(clean)} ADR  (đã loại {len(removed)})")
    print(f"Chi tiết: {REPORT_FILE}   |   Bản gốc: {BACKUP_FILE}")
    counts = {}
    for x in clean:
        counts[x["group"]] = counts.get(x["group"], 0) + 1
    print(f"Số repo duy nhất còn lại: {len({x['repo'] for x in clean})}")
    print("Phân bố theo nhóm sau khi loại trùng:")
    for g in sorted(counts):
        print(f"  {g}: {counts[g]}" + ("  << dưới 15" if counts[g] < 15 else ""))


if __name__ == "__main__":
    main()
