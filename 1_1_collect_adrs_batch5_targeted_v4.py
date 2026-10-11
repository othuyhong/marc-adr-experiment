"""
1_1_collect_adrs_batch5_targeted_v4.py
BƯỚC 1.1 (batch 5) - Tìm bổ sung ADR cho các nhóm G1-G7 đang có dưới TARGET_PER_GROUP ADR,
bằng GitHub Code Search với các truy vấn riêng cho từng nhóm.

Cách hoạt động:
  1. Đọc labeled_adrs.json và đếm nhóm nào đang thiếu.
  2. Với mỗi nhóm thiếu, chạy các truy vấn Code Search của nhóm đó.
  3. Mỗi file tìm được: trích Context/Decision, chấm điểm bằng bộ từ khóa của
     1_3_label_groups_auto.py, và chỉ nhận khi máy tự gán được (đủ điểm, hơn nhóm nhì rõ ràng)
     vào một nhóm đang thiếu. Bản ghi nhận vào có label_source = "search".
  4. Ghi nối thêm vào labeled_adrs.json; URL đã loại được nhớ trong skipped_labels.json.

Chịu được rớt mạng: tự đợi mạng có lại rồi thử tiếp, và lưu tiến độ sau mỗi truy vấn.
Chạy lại sẽ bỏ qua các file đã xử lý.

Yêu cầu:
  - File 1_3_label_groups_auto.py nằm cùng thư mục.
  - Token GitHub đặt trong biến môi trường GITHUB_TOKEN (không ghi token vào file này).

Cách chạy (PowerShell):
    $env:GITHUB_TOKEN="<token của bạn>"
    python 1_1_collect_adrs_batch5_targeted_v4.py

Chạy xong thì chạy 1_3b_dedupe.py để loại trùng trước các bước tiếp theo.
"""
import importlib
import json
import os
import re
import time

import requests
from github import Auth, Github, GithubException, RateLimitExceededException

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise SystemExit("Chưa set GITHUB_TOKEN.")

try:
    lab = importlib.import_module("1_3_label_groups_auto")
except ModuleNotFoundError:
    raise SystemExit("Cần file 1_3_label_groups_auto.py nằm cùng thư mục với script này.")

g = Github(auth=Auth.Token(GITHUB_TOKEN), per_page=100, timeout=30)

NET_ERRORS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)

LABELED_FILE = "labeled_adrs.json"
SKIP_FILE = "skipped_labels.json"
TARGET_PER_GROUP = 15
MAX_RESULTS_PER_QUERY = 200

SEARCH_QUERIES = {
    "G1_ServiceDecomposition": [
        '"bounded context" "## Decision" extension:md',
        '"monolith" "microservices" "## Decision" extension:md',
        '"service boundaries" "## Context" extension:md',
        '"domain-driven design" "## Decision" extension:md',
        '"decompose" "service" "## Decision" extension:md',
        '"modular monolith" extension:md',
        '"monolith" "## Context" extension:md',
        '"bounded contexts" "microservices" extension:md',
        '"strangler" "monolith" extension:md',
    ],
    "G2_Communication": [
        '"event-driven" "## Decision" extension:md',
        '"message queue" "## Context" extension:md',
        '"grpc" "## Decision" extension:md',
        '"asynchronous" "microservices" "## Decision" extension:md',
    ],
    "G3_DataManagement": [
        '"database per service" extension:md',
        '"saga" "## Decision" extension:md',
        '"event sourcing" "## Decision" extension:md',
        '"cqrs" "## Context" extension:md',
    ],
    "G4_APIGatewayDiscovery": [
        '"API gateway" "## Decision" extension:md',
        '"service discovery" "## Decision" extension:md',
        '"rate limiting" "## Context" extension:md',
        '"backend for frontend" extension:md',
        '"API gateway" "## Context" extension:md',
        '"service discovery" "## Context" extension:md',
        '"load balancer" "## Decision" extension:md',
        '"routing" "microservices" "## Decision" extension:md',
    ],
    "G5_DeploymentOrchestration": [
        '"kubernetes" "## Decision" extension:md',
        '"helm" "## Context" extension:md',
        '"docker" "microservices" "## Decision" extension:md',
    ],
    "G6_Resilience": [
        '"circuit breaker" "## Decision" extension:md',
        '"circuit breaker" "## Context" extension:md',
        '"retry" "fallback" "## Decision" extension:md',
        '"resilience" "## Decision" extension:md',
        '"timeout" "retry" "## Context" extension:md',
        '"bulkhead" extension:md',
        '"fault tolerance" "## Decision" extension:md',
    ],
    "G7_ObservabilitySecurity": [
        '"observability" "## Decision" extension:md',
        '"distributed tracing" extension:md',
        '"oauth" "## Decision" extension:md',
        '"secret management" extension:md',
    ],
}

CONTEXT_PATTERNS = [
    r"##\s*Context(?:\s+and\s+Problem\s+Statement)?\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"\*\*Context\*\*\s*\n+(.*?)(?=\n\*\*|\Z)",
    r"##\s*Decision\s+Drivers?\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"#\s*Context\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Problem\s+Statement\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Background\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Motivation\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
]
DECISION_PATTERNS = [
    r"##\s*Decision(?:\s+Outcome)?\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"\*\*Decision\*\*\s*\n+(.*?)(?=\n\*\*|\Z)",
    r"##\s*Chosen\s+Option\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"#\s*Decision\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Proposal\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
    r"##\s*Solution\s*\n+(.*?)(?=\n#{1,3}\s|\Z)",
]


def extract(content):
    ctx = dec = None
    for pat in CONTEXT_PATTERNS:
        m = re.search(pat, content, re.DOTALL | re.IGNORECASE)
        if m:
            ctx = m.group(1).strip()
            break
    for pat in DECISION_PATTERNS:
        m = re.search(pat, content, re.DOTALL | re.IGNORECASE)
        if m:
            dec = m.group(1).strip()
            break
    return ctx, dec


def classify(content, counts):
    """Trả về (kết_quả, lý_do). kết_quả = (nhóm, điểm, margin, context, decision) nếu nhận,
    ngược lại None kèm lý do: 'no_struct' | 'low_score' | 'group_full'."""
    ctx, dec = extract(content)
    if not (ctx and len(ctx) >= 100 and dec):
        return None, "no_struct"
    action, top_g, top_s, margin = lab.decide(lab.score_groups(ctx + " " + dec))
    if action != "assign":
        return None, "low_score"
    if counts.get(top_g, 0) >= TARGET_PER_GROUP:
        return None, "group_full"
    return (top_g, top_s, margin, ctx, dec), "ok"


def wait_for_network(max_minutes=30):
    """Đợi đến khi gọi được api.github.com. Trả về False nếu quá max_minutes vẫn không được."""
    for i in range(max_minutes * 2):
        try:
            requests.get("https://api.github.com", timeout=10)
            return True
        except NET_ERRORS:
            if i == 0:
                print("    [MẤT MẠNG] đang đợi kết nối lại (kiểm tra Wi-Fi/VPN)...")
            time.sleep(30)
    return False


def safe_search(query):
    """Trả về các file khớp truy vấn. Rớt mạng giữa chừng -> đợi mạng rồi thử lại cả truy vấn."""
    results = []
    for attempt in range(3):
        results = []
        try:
            for i, item in enumerate(g.search_code(query=query)):
                if i >= MAX_RESULTS_PER_QUERY:
                    break
                results.append(item)
            return results
        except RateLimitExceededException:
            print("    [RATE LIMIT] đợi 60s rồi thử lại...")
            time.sleep(60)
        except GithubException as e:
            print(f"    [LỖI SEARCH] mã {e.status}: {e.data}")
            return results
        except NET_ERRORS:
            if not wait_for_network():
                raise SystemExit("Mất mạng quá 30 phút. Tiến độ đã lưu - chạy lại script khi có mạng.")
            print("    [CÓ MẠNG LẠI] thử lại truy vấn...")
    return results


def fetch_content(item):
    """Tải nội dung file; rớt mạng thì đợi mạng và thử lại một lần."""
    try:
        return item.decoded_content.decode("utf-8", errors="ignore")
    except NET_ERRORS:
        if not wait_for_network():
            raise SystemExit("Mất mạng quá 30 phút. Tiến độ đã lưu - chạy lại script khi có mạng.")
        return item.decoded_content.decode("utf-8", errors="ignore")


def save_progress(labeled, skipped):
    with open(LABELED_FILE, "w", encoding="utf-8") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)
    with open(SKIP_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(set(skipped)), f, ensure_ascii=False, indent=2)


def main():
    if not os.path.exists(LABELED_FILE):
        raise SystemExit(f"Không thấy {LABELED_FILE}. Chạy 1_3_label_groups_auto.py trước.")
    with open(LABELED_FILE, "r", encoding="utf-8") as f:
        labeled = json.load(f)
    skipped = []
    if os.path.exists(SKIP_FILE):
        with open(SKIP_FILE, "r", encoding="utf-8") as f:
            skipped = json.load(f)

    existing = {x["url"] for x in labeled} | set(skipped)
    counts = {}
    for x in labeled:
        counts[x["group"]] = counts.get(x["group"], 0) + 1

    needy = [gn for gn in lab.GROUP_NAMES if counts.get(gn, 0) < TARGET_PER_GROUP]
    print("Phân bố hiện tại:")
    for gn in lab.GROUP_NAMES:
        print(f"  {gn}: {counts.get(gn, 0)}" + ("  << sẽ tìm thêm" if gn in needy else ""))
    if not needy:
        print(f"Mọi nhóm đã đủ {TARGET_PER_GROUP} ADR, không cần tìm thêm.")
        return

    added = 0
    try:
        for target in needy:
            for query in SEARCH_QUERIES.get(target, []):
                if counts.get(target, 0) >= TARGET_PER_GROUP:
                    break
                print(f"\n[TÌM {target}] {query}")
                items = safe_search(query)
                st = {"đã có": 0, "không trích được": 0, "điểm thấp/nhập nhằng": 0,
                      "nhóm đã đủ": 0, "NHẬN": 0}
                for item in items:
                    if all(counts.get(gn, 0) >= TARGET_PER_GROUP for gn in needy):
                        break
                    url = item.html_url
                    if url in existing:
                        st["đã có"] += 1
                        continue
                    try:
                        content = fetch_content(item)
                    except RateLimitExceededException:
                        print("    [RATE LIMIT khi tải file] đợi 30s...")
                        time.sleep(30)
                        continue
                    except Exception as e:
                        print(f"    lỗi đọc {item.path}: {e}")
                        continue
                    res, why = classify(content, counts)
                    existing.add(url)
                    if res is None:
                        if why == "no_struct":
                            st["không trích được"] += 1
                            skipped.append(url)
                        elif why == "low_score":
                            st["điểm thấp/nhập nhằng"] += 1
                            skipped.append(url)
                        else:
                            st["nhóm đã đủ"] += 1
                        continue
                    grp, score, margin, ctx, dec = res
                    labeled.append({
                        "repo": item.repository.full_name, "file": item.name,
                        "path": item.path, "url": url, "context": ctx, "decision": dec,
                        "context_length": len(ctx), "group": grp,
                        "label_source": "search", "auto_score": score, "auto_margin": margin,
                    })
                    counts[grp] = counts.get(grp, 0) + 1
                    added += 1
                    st["NHẬN"] += 1
                    print(f"    + {grp}: {item.repository.full_name}/{item.name}")
                    time.sleep(0.3)
                print(f"    -> {len(items)} file | " + " | ".join(f"{k}: {v}" for k, v in st.items()))
                save_progress(labeled, skipped)
                time.sleep(6)   # Code Search API chỉ cho khoảng 10 lượt/phút
    except KeyboardInterrupt:
        print("\nĐã nhận Ctrl+C, đang lưu những gì đã tìm được...")
    finally:
        save_progress(labeled, skipped)

    print(f"\n=== TÓM TẮT: thêm {added} ADR, tổng {len(labeled)} ===")
    for gn in lab.GROUP_NAMES:
        c = counts.get(gn, 0)
        print(f"  {gn}: {c}" + ("  << VẪN THIẾU" if c < TARGET_PER_GROUP else ""))
    print("Nhớ chạy 1_3b_dedupe.py trước khi tách test/RAG.")


if __name__ == "__main__":
    main()
