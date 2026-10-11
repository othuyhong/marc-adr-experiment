"""
1_3_label_groups_auto.py
Gán nhãn 7 nhóm G1-G7 theo kiểu "tự gán khi chắc chắn, chỉ hỏi khi không chắc".

Quy tắc cho mỗi ADR (điểm = số lần từ khóa của nhóm xuất hiện trong Context + Decision):
  - Tất cả nhóm điểm = 0                       -> tự BỎ QUA (không thuộc microservices)
  - Nhóm cao nhất >= AUTO_MIN_SCORE điểm
    VÀ hơn nhóm nhì >= AUTO_MIN_MARGIN điểm    -> tự GÁN, không hỏi
  - Còn lại (điểm thấp hoặc hai nhóm sát nhau) -> HỎI bạn

Khi hỏi, bạn gõ:
  Enter = đồng ý với đề xuất | 1-7 = chọn nhóm khác | s = bỏ qua | q = lưu và thoát

Cách chạy (cùng thư mục với raw_adrs.json):
    python 1_3_label_groups_auto.py

Có thể chạy tiếp sau bản cũ 1_3_label_groups.py: những ADR đã nằm trong
labeled_adrs.json được giữ nguyên, chỉ làm phần còn lại.

Kết quả:
  - labeled_adrs.json     : ADR đã gán nhãn (thêm trường label_source = auto/manual)
  - skipped_labels.json   : URL các ADR đã bỏ qua (lần chạy sau không hỏi lại)
  - auto_labeled_review.csv : danh sách ADR do MÁY tự gán, để bạn rà soát lại bằng mắt
"""
import csv
import json
import os
import re

IN_FILE = "raw_adrs.json"
OUT_FILE = "labeled_adrs.json"
SKIP_FILE = "skipped_labels.json"
REVIEW_FILE = "auto_labeled_review.csv"

# Ngưỡng tự gán (chỉnh nếu muốn máy hỏi nhiều hơn / ít hơn)
AUTO_MIN_SCORE = 3
AUTO_MIN_MARGIN = 2

# Từ khóa khớp theo ĐẦU TỪ (ví dụ "decompos" khớp decompose/decomposition,
# "auth" KHÔNG còn dùng vì sẽ khớp nhầm "author").
GROUP_KEYWORDS = {
    "G1_ServiceDecomposition": ["service boundary", "decompos", "bounded context",
                                 "monolith", "split service", "domain-driven"],
    "G2_Communication": ["sync", "async", "message queue", "event-driven", "rest api",
                          "grpc", "pub/sub", "kafka", "nats", "rabbitmq"],
    "G3_DataManagement": ["database per service", "saga", "cqrs", "event sourcing",
                           "schema", "outbox", "transactional", "cdc"],
    "G4_APIGatewayDiscovery": ["api gateway", "service discovery", "routing",
                                "load balancer", "rate limit", "backend for frontend"],
    "G5_DeploymentOrchestration": ["kubernetes", "helm", "docker", "container",
                                    "deployment", "scaling", "orchestration", "ci/cd"],
    "G6_Resilience": ["circuit breaker", "retry", "bulkhead", "timeout", "resilience",
                       "fallback", "fault tolerance"],
    "G7_ObservabilitySecurity": ["logging", "tracing", "metrics", "observability",
                                  "authentication", "authorization", "authn", "authz",
                                  "jwt", "oauth", "secret management", "vault"],
}
GROUP_NAMES = list(GROUP_KEYWORDS.keys())
PATTERNS = {
    g: [re.compile(r"(?<![a-z0-9])" + re.escape(kw)) for kw in kws]
    for g, kws in GROUP_KEYWORDS.items()
}


def score_groups(text):
    text = text.lower()
    return {g: sum(len(p.findall(text)) for p in pats) for g, pats in PATTERNS.items()}


def decide(scores):
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (top_g, top_s), (_, second_s) = ranked[0], ranked[1]
    if top_s == 0:
        return "skip", top_g, top_s, 0
    if top_s >= AUTO_MIN_SCORE and (top_s - second_s) >= AUTO_MIN_MARGIN:
        return "assign", top_g, top_s, top_s - second_s
    return "ask", top_g, top_s, top_s - second_s


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_all(labeled, skipped):
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)
    with open(SKIP_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(skipped), f, ensure_ascii=False, indent=2)


def write_review(labeled):
    autos = [x for x in labeled if x.get("label_source") == "auto"]
    with open(REVIEW_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group", "auto_score", "margin_over_2nd", "repo", "file", "url",
                    "context_preview"])
        for x in sorted(autos, key=lambda x: x["group"]):
            w.writerow([x["group"], x.get("auto_score", ""), x.get("auto_margin", ""),
                        x["repo"], x["file"], x["url"],
                        (x.get("context") or "")[:150].replace("\n", " ")])
    return len(autos)


def main():
    if not os.path.exists(IN_FILE):
        raise SystemExit(f"Không thấy {IN_FILE}. Chạy các script thu thập (1_1_...) trước.")

    data = load_json(IN_FILE, [])
    labeled = load_json(OUT_FILE, [])
    skipped = set(load_json(SKIP_FILE, []))
    done = {x["url"] for x in labeled}

    todo = [x for x in data if x["url"] not in done and x["url"] not in skipped]
    print(f"Tổng ADR thô: {len(data)} | đã gán: {len(labeled)} | đã bỏ qua: {len(skipped)} "
          f"| còn phải xử lý: {len(todo)}")

    n_auto = n_skip_auto = n_manual = n_skip_manual = 0
    quit_early = False

    try:
        for idx, item in enumerate(todo, 1):
            text = (item.get("context") or "") + " " + (item.get("decision") or "")
            scores = score_groups(text)
            action, top_g, top_s, margin = decide(scores)

            if action == "skip":
                skipped.add(item["url"])
                n_skip_auto += 1
            elif action == "assign":
                item["group"] = top_g
                item["label_source"] = "auto"
                item["auto_score"] = top_s
                item["auto_margin"] = margin
                labeled.append(item)
                n_auto += 1
            else:
                print("\n" + "=" * 70)
                print(f"[{idx}/{len(todo)}] Repo: {item['repo']}  |  File: {item['file']}")
                print(f"Context: {(item.get('context') or '')[:300]}...")
                print(f"Decision: {(item.get('decision') or '')[:200]}...")
                print("-" * 70)
                for i, g in enumerate(GROUP_NAMES, 1):
                    mark = " <== đề xuất" if g == top_g else ""
                    print(f"  {i}. {g} (score={scores[g]}){mark}")
                try:
                    choice = input(f"Chọn nhóm [Enter={GROUP_NAMES.index(top_g) + 1}, "
                                   f"1-7, s=skip, q=quit]: ").strip().lower()
                except EOFError:
                    choice = "q"

                if choice == "q":
                    quit_early = True
                    break
                if choice == "s":
                    skipped.add(item["url"])
                    n_skip_manual += 1
                elif choice == "" or (choice.isdigit() and 1 <= int(choice) <= 7):
                    item["group"] = top_g if choice == "" else GROUP_NAMES[int(choice) - 1]
                    item["label_source"] = "manual"
                    labeled.append(item)
                    n_manual += 1
                else:
                    print("Lựa chọn không hợp lệ, bỏ qua ADR này (sẽ hỏi lại lần sau).")
                    continue
                save_all(labeled, skipped)

            if idx % 25 == 0:
                save_all(labeled, skipped)
    except KeyboardInterrupt:
        print("\nĐã nhận Ctrl+C, đang lưu tiến độ...")
        quit_early = True

    save_all(labeled, skipped)
    n_review = write_review(labeled)

    print("\n=== TÓM TẮT LẦN CHẠY NÀY ===")
    print(f"  Máy tự gán nhãn     : {n_auto}")
    print(f"  Máy tự bỏ qua (0đ)  : {n_skip_auto}")
    print(f"  Bạn gán thủ công    : {n_manual}")
    print(f"  Bạn bỏ qua          : {n_skip_manual}")
    if quit_early:
        print("  (Chưa xử lý hết - chạy lại script này để làm tiếp.)")

    counts = {}
    for x in labeled:
        counts[x["group"]] = counts.get(x["group"], 0) + 1
    print(f"\nTổng ADR đã gán nhãn: {len(labeled)}  ->  lưu tại {OUT_FILE}")
    print("Phân bố theo nhóm:")
    for g in GROUP_NAMES:
        c = counts.get(g, 0)
        print(f"  {g}: {c}" + ("  << CẦN THÊM (mục tiêu >=15)" if c < 15 else ""))
    print(f"\nĐã xuất {n_review} ADR do máy tự gán vào {REVIEW_FILE} - mở bằng Excel để rà soát.")


if __name__ == "__main__":
    main()
