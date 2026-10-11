"""
1_1_collect_adrs_batch3.py
BƯỚC 1.1 (BATCH 3) — Thêm 15 repo còn lại (Tier 3 + Tier 5 chưa thử) để đạt mốc ~120 ADR.
Chạy SAU KHI đã có batch 1 (v2) + batch 2, sẽ nối tiếp vào raw_adrs.json hiện có.

Cách chạy:
    python 1_1_collect_adrs_batch3.py
"""
import os
import re
import csv
import json
import time
from github import Github, GithubException

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise SystemExit("Chưa set GITHUB_TOKEN.")

g = Github(GITHUB_TOKEN)

REPOS_BATCH3 = [
    "Netflix/conductor",
    "GoogleCloudPlatform/microservices-demo",
    "thingsboard/thingsboard",
    "apache/skywalking",
    "grafana/grafana",
    "aws-samples/aws-microservices-deploy-options",
    "buildkite/agent",
    "sourcegraph/sourcegraph",
    "open-telemetry/opentelemetry-collector",
    "helm/helm",
    "confluentinc/schema-registry",
    "istio/istio",
    "kubernetes/kubernetes",
    "spiffe/spire",                    # bổ sung ngoài danh sách gốc: identity/security microservices, có docs/decisions
    "googleapis/google-cloud-go",      # bổ sung: monorepo lớn nhưng có design docs rải rác
]

MAX_TREE_ENTRIES = 80000  # tăng giới hạn vì batch này có vài monorepo lớn (kubernetes, istio)

PATH_KEYWORDS = ["adr", "decision", "rfc", "rationale"]
EXCLUDE_KEYWORDS = ["readme", "template", "index.md", "changelog"]

CONTEXT_PATTERNS = [
    r"##\s*Context(?:\s+and\s+Problem\s+Statement)?\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"\*\*Context\*\*\s*\n+(.*?)(?=\n\*\*|\Z)",
    r"##\s*Decision\s+Drivers?\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"###\s*Context\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"#\s*Context\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Problem\s+Statement\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Background\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Motivation\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Problem\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",               # mới: batch 3
    r"##\s*Summary\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",                # mới: nhiều RFC/KEP-style docs dùng Summary làm phần mở đầu bối cảnh
]
DECISION_PATTERNS = [
    r"##\s*Decision(?:\s+Outcome)?\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"\*\*Decision\*\*\s*\n+(.*?)(?=\n\*\*|\Z)",
    r"###\s*Decision\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Chosen\s+Option\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"#\s*Decision\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Proposal\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Solution\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Rationale\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",              # mới: batch 3
    r"##\s*Design\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",                 # mới: KEP/design-doc style
]


def find_adr_files_full_scan(repo):
    found = []
    try:
        default_branch = repo.default_branch
        tree = repo.get_git_tree(sha=default_branch, recursive=True)
    except GithubException as e:
        print(f"    [LỖI TREE] {repo.full_name}: {e}")
        return found

    if tree.raw_data.get("truncated"):
        print(f"    [CẢNH BÁO] Cây thư mục {repo.full_name} bị GitHub cắt bớt (repo quá lớn) — "
              f"kết quả có thể thiếu, nhưng vẫn quét phần đã lấy được.")

    entries = tree.tree[:MAX_TREE_ENTRIES]
    for item in entries:
        if item.type != "blob":
            continue
        path_lower = item.path.lower()
        if not path_lower.endswith(".md"):
            continue
        if not any(kw in path_lower for kw in PATH_KEYWORDS):
            continue
        if any(kw in path_lower for kw in EXCLUDE_KEYWORDS):
            continue
        found.append({
            "path": item.path,
            "name": item.path.split("/")[-1],
            "url": f"https://github.com/{repo.full_name}/blob/{default_branch}/{item.path}",
        })
    return found


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


def main():
    dataset = []
    if os.path.exists("raw_adrs.json"):
        with open("raw_adrs.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
    existing_urls = {item["url"] for item in dataset}
    start_count = len(dataset)

    log_rows = []
    skipped_rows = []

    for repo_name in REPOS_BATCH3:
        print(f"[SCAN] {repo_name}")
        row = {"repo": repo_name, "md_files_matched": 0,
               "valid_context_decision": 0, "status": "ok"}
        try:
            repo = g.get_repo(repo_name)
            candidate_files = find_adr_files_full_scan(repo)
            row["md_files_matched"] = len(candidate_files)
            print(f"    -> tìm thấy {len(candidate_files)} file .md khả nghi")

            for f in candidate_files:
                if f["url"] in existing_urls:
                    continue
                try:
                    file_content = repo.get_contents(f["path"])
                    content_str = file_content.decoded_content.decode("utf-8", errors="ignore")
                    extracted = extract_context_decision(content_str)

                    if extracted["context"] and len(extracted["context"]) >= 100 and extracted["decision"]:
                        dataset.append({
                            "repo": repo_name,
                            "file": f["name"],
                            "path": f["path"],
                            "url": f["url"],
                            "context": extracted["context"],
                            "decision": extracted["decision"],
                            "context_length": len(extracted["context"]),
                        })
                        row["valid_context_decision"] += 1
                        existing_urls.add(f["url"])
                    else:
                        skipped_rows.append({
                            "repo": repo_name, "path": f["path"],
                            "reason": "context/decision không khớp pattern nào"
                        })
                    time.sleep(0.1)
                except Exception as e:
                    skipped_rows.append({"repo": repo_name, "path": f["path"], "reason": str(e)})

        except GithubException as e:
            row["status"] = f"repo_error: {e}"
            print(f"  [LỖI REPO] {repo_name}: {e}")

        log_rows.append(row)

        # dừng sớm nếu đã đạt đủ 120 ADR để tiết kiệm thời gian/quota — comment dòng dưới nếu muốn quét hết
        if len(dataset) >= 100000:
            print(f"\n[ĐỦ MỤC TIÊU] Đã đạt {len(dataset)} ADR (>=120) — dừng sớm, không cần quét nốt danh sách.")
            break

    with open("raw_adrs.json", "w", encoding="utf-8") as fp:
        json.dump(dataset, fp, ensure_ascii=False, indent=2)

    write_header = not os.path.exists("collection_log.csv")
    with open("collection_log.csv", "a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["repo", "md_files_matched",
                                                 "valid_context_decision", "status"])
        if write_header:
            writer.writeheader()
        writer.writerows(log_rows)

    write_header2 = not os.path.exists("skipped_files_log.csv")
    with open("skipped_files_log.csv", "a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["repo", "path", "reason"])
        if write_header2:
            writer.writeheader()
        writer.writerows(skipped_rows)

    print("\n=== TÓM TẮT BATCH 3 ===")
    for r in log_rows:
        print(f"{r['repo']:55s} khớp={r['md_files_matched']:4d}  hợp lệ={r['valid_context_decision']:3d}")
    print(f"\nADR trước batch 3: {start_count}  ->  Tổng sau batch 3: {len(dataset)}")
    if len(dataset) < 120:
        print("Chưa đạt 120 — cân nhắc bỏ comment dòng dừng sớm và bổ sung thêm repo nếu cần.")
    else:
        print("Đã đạt mục tiêu tối thiểu 120 ADR — có thể chuyển sang Bước 1.2/1.3.")


if __name__ == "__main__":
    main()
