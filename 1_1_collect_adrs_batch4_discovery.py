"""
1_1_collect_adrs_batch4_discovery.py
BƯỚC 1.1 (BATCH 4 — CHIẾN THUẬT MỚI) — Thay vì đoán tên repo, dùng GitHub Code Search API
để TỰ ĐỘNG tìm các file trên toàn GitHub có định dạng ADR thật (chứa "## Context"/"## Decision"
hoặc nằm trong đường dẫn adr/decisions). Hiệu quả hơn nhiều so với đoán tên repo mù.

Lưu ý quan trọng: GitHub Code Search API giới hạn RẤT chặt (khoảng 10 request/phút kể cả có token),
nên script này chạy chậm hơn (có sleep giữa các lần gọi) — kiên nhẫn đợi, đừng Ctrl+C giữa chừng.

Cách chạy:
    python 1_1_collect_adrs_batch4_discovery.py
"""
import os
import re
import csv
import json
import time
from github import Github, GithubException, RateLimitExceededException

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise SystemExit("Chưa set GITHUB_TOKEN.")

g = Github(GITHUB_TOKEN)

# Các câu truy vấn tìm kiếm code — mỗi câu nhắm vào một cách viết ADR khác nhau
SEARCH_QUERIES = [
    '"## Context" "## Decision" extension:md path:adr',
    '"## Context" "## Decision" extension:md path:decisions',
    '"Architecture Decision Record" extension:md path:docs',
    '"## Decision Drivers" extension:md',
    '"## Considered Options" extension:md',
]

MAX_RESULTS_PER_QUERY = 60      # đủ để tìm nhiều repo mới mà không tốn quá nhiều quota search
TARGET_TOTAL = 100000

CONTEXT_PATTERNS = [
    r"##\s*Context(?:\s+and\s+Problem\s+Statement)?\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"\*\*Context\*\*\s*\n+(.*?)(?=\n\*\*|\Z)",
    r"##\s*Decision\s+Drivers?\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"###\s*Context\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"#\s*Context\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Problem\s+Statement\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Background\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Motivation\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
]
DECISION_PATTERNS = [
    r"##\s*Decision(?:\s+Outcome)?\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"\*\*Decision\*\*\s*\n+(.*?)(?=\n\*\*|\Z)",
    r"###\s*Decision\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Chosen\s+Option\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"#\s*Decision\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Proposal\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Solution\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
]


def extract_context_decision(content_str):
    context, decision = None, None
    for pat in CONTEXT_PATTERNS:
        m = re.search(pat, content_str, re.DOTALL | re.IGNORECASE)
        if m:
            context = m.group(1).strip()
            break
    for pat in DECISION_PATTERNS:
        m = re.search(pat, content_str, re.DOTALL | re.IGNORECASE)
        if m:
            decision = m.group(1).strip()
            break
    return {"context": context, "decision": decision}


def safe_search(query, max_results):
    """Gọi search_code với retry khi bị rate-limit."""
    results = []
    try:
        paginated = g.search_code(query=query)
        for i, item in enumerate(paginated):
            if i >= max_results:
                break
            results.append(item)
            time.sleep(2.5)  # tôn trọng giới hạn ~10 req/phút của Code Search API
    except RateLimitExceededException:
        print("    [RATE LIMIT] Search API bị giới hạn, đợi 60s...")
        time.sleep(60)
    except GithubException as e:
        print(f"    [LỖI SEARCH] query='{query}': {e}")
    return results


def main():
    dataset = []
    if os.path.exists("raw_adrs.json"):
        with open("raw_adrs.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
    existing_urls = {item["url"] for item in dataset}
    start_count = len(dataset)

    skipped_rows = []
    discovered_repos = set()

    for query in SEARCH_QUERIES:
        if len(dataset) >= TARGET_TOTAL:
            print(f"\n[ĐỦ MỤC TIÊU] Đã đạt {len(dataset)} ADR — dừng sớm.")
            break

        print(f"\n[TÌM KIẾM] {query}")
        items = safe_search(query, MAX_RESULTS_PER_QUERY)
        print(f"    -> {len(items)} file khớp truy vấn")

        for item in items:
            if len(dataset) >= TARGET_TOTAL:
                break

            repo_name = item.repository.full_name
            file_url = item.html_url
            discovered_repos.add(repo_name)

            if file_url in existing_urls:
                continue

            try:
                content_str = item.decoded_content.decode("utf-8", errors="ignore")
                extracted = extract_context_decision(content_str)

                if extracted["context"] and len(extracted["context"]) >= 100 and extracted["decision"]:
                    dataset.append({
                        "repo": repo_name,
                        "file": item.name,
                        "path": item.path,
                        "url": file_url,
                        "context": extracted["context"],
                        "decision": extracted["decision"],
                        "context_length": len(extracted["context"]),
                    })
                    existing_urls.add(file_url)
                else:
                    skipped_rows.append({"repo": repo_name, "path": item.path,
                                          "reason": "context/decision không khớp pattern nào"})
                time.sleep(0.3)
            except RateLimitExceededException:
                print("    [RATE LIMIT] Content API bị giới hạn, đợi 30s...")
                time.sleep(30)
            except Exception as e:
                skipped_rows.append({"repo": repo_name, "path": item.path, "reason": str(e)})

    with open("raw_adrs.json", "w", encoding="utf-8") as fp:
        json.dump(dataset, fp, ensure_ascii=False, indent=2)

    write_header2 = not os.path.exists("skipped_files_log.csv")
    with open("skipped_files_log.csv", "a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["repo", "path", "reason"])
        if write_header2:
            writer.writeheader()
        writer.writerows(skipped_rows)

    with open("discovered_repos.txt", "w", encoding="utf-8") as fp:
        for r in sorted(discovered_repos):
            fp.write(r + "\n")

    print("\n=== TÓM TẮT BATCH 4 (DISCOVERY) ===")
    print(f"ADR trước batch 4: {start_count}  ->  Tổng sau batch 4: {len(dataset)}")
    print(f"Số repo mới được phát hiện qua search: {len(discovered_repos)} (xem discovered_repos.txt)")
    if len(dataset) >= 120:
        print("Đã đạt mục tiêu tối thiểu 120 ADR — chuyển sang Bước 1.3 (gán nhãn nhóm).")
    else:
        print("Vẫn chưa đủ 120 — chạy lại script này thêm 1 lần nữa (search sẽ tiếp tục từ trạng thái mới, "
              "vì không giới hạn nghiêm ngặt số ADR mỗi query có thể trả về nhiều hơn ở lần chạy tiếp).")


if __name__ == "__main__":
    main()
