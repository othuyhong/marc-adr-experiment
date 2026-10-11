"""
4_repo_count.py
Giải quyết R3#12: "The manuscript should report the exact number of unique
GitHub repositories rather than '44+'."

Cách chạy (từ thư mục chứa labeled_adrs.json):
    python 4_repo_count.py

Kết quả: in ra số repo duy nhất CHÍNH XÁC, kèm danh sách — copy số này
để thay thế mọi chỗ ghi "44+" trong main.tex (Abstract, Introduction,
Section 4.2, Data Availability Statement).
"""
import json

with open("labeled_adrs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

repos = sorted(set(item["repo"] for item in data))

print(f"TỔNG SỐ REPO DUY NHẤT: {len(repos)}")
print(f"TỔNG SỐ ADR: {len(data)}")
print(f"\nDanh sách {len(repos)} repo:")
for r in repos:
    count = sum(1 for item in data if item["repo"] == r)
    print(f"  {r}: {count} ADR")

print(f"\n>>> Copy con số {len(repos)} này để thay thế '44+' trong main.tex <<<")
