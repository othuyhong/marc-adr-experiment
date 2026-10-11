"""
1_4_split_dataset.py
BƯỚC 1.4 — Tách gold test set (đánh giá) và RAG pool (knowledge base), không trùng nhau.

THAY ĐỔI SO VỚI BẢN CŨ (theo góp ý R3#9, #10, #11, #16):
  - BỎ ưu tiên ADR có "Considered Options". Test set giờ là mẫu NGẪU NHIÊN PHÂN TẦNG
    theo nhóm G1-G7, seed cố định -> tái lập được, không thiên về ADR viết kỹ.
  - Mỗi nhóm không bị rút quá MAX_TEST_FRACTION vào test set, để RAG pool vẫn còn
    ví dụ của nhóm đó.
  - ADR có test_eligible = false (Decision dài hơn giới hạn output của model, do
    1_3d_finalize_dataset.py gắn) chỉ nằm trong RAG pool, không vào test set.
  - Kết quả không phụ thuộc thứ tự bản ghi trong labeled_adrs.json (sắp theo URL trước).
  - Báo cáo số mẫu test theo từng nhóm, số repo, và với mỗi ADR test có bao nhiêu ADR
    cùng repo nằm trong RAG pool (những ADR này sẽ bị loại lúc retrieve ở bước 2).

Cách chạy (sau 1_3b_dedupe.py):
    python 1_4_split_dataset.py

Kết quả:
    - gold_test_set.json
    - rag_pool.json
    - split_summary.csv     : số ADR test / pool theo nhóm  (dùng cho bảng trong bài)
    - split_repo_overlap.csv: ADR test nào có ADR cùng repo trong pool
"""
import csv
import json
import random
from collections import Counter, defaultdict

SEED = 42
IN_FILE = "labeled_adrs.json"
TEST_SIZE = 35
MAX_TEST_FRACTION = 0.5     # không rút quá 50% một nhóm vào test set


def main():
    with open(IN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    urls = [x["url"] for x in data]
    if len(set(urls)) != len(urls):
        raise SystemExit("labeled_adrs.json còn URL trùng nhau - chạy 1_3b_dedupe.py trước.")
    if len(data) <= TEST_SIZE:
        raise SystemExit(f"Chỉ có {len(data)} ADR, không đủ để tách {TEST_SIZE} ADR test.")

    rng = random.Random(SEED)
    by_group = defaultdict(list)
    for item in sorted(data, key=lambda x: x["url"]):
        by_group[item["group"]].append(item)
    groups = sorted(by_group)

    # Hạn mức mỗi nhóm = chia đều, nhưng không vượt MAX_TEST_FRACTION của nhóm
    quota = TEST_SIZE // len(groups)
    cap = {g: int(len(by_group[g]) * MAX_TEST_FRACTION) for g in groups}
    test_set, leftovers = [], []
    picked_per_group = Counter()
    for g in groups:
        items = [x for x in by_group[g] if x.get("test_eligible", True)]
        rng.shuffle(items)
        n = min(quota, cap[g], len(items))
        test_set.extend(items[:n])
        leftovers.extend(items[n:])
        picked_per_group[g] = n

    # Lấy bù ngẫu nhiên cho đủ TEST_SIZE, vẫn tôn trọng trần của từng nhóm
    rng.shuffle(leftovers)
    for item in leftovers:
        if len(test_set) >= TEST_SIZE:
            break
        g = item["group"]
        if picked_per_group[g] < cap[g]:
            test_set.append(item)
            picked_per_group[g] += 1

    if len(test_set) < TEST_SIZE:
        print(f"[CẢNH BÁO] Chỉ tách được {len(test_set)}/{TEST_SIZE} ADR test vì trần "
              f"{MAX_TEST_FRACTION:.0%} mỗi nhóm. Thu thập thêm ADR hoặc giảm TEST_SIZE.")

    test_urls = {x["url"] for x in test_set}
    rag_pool = [x for x in data if x["url"] not in test_urls]
    assert not test_urls & {x["url"] for x in rag_pool}

    with open("gold_test_set.json", "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    with open("rag_pool.json", "w", encoding="utf-8") as f:
        json.dump(rag_pool, f, ensure_ascii=False, indent=2)

    pool_per_group = Counter(x["group"] for x in rag_pool)
    with open("split_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group", "total", "test_set", "rag_pool"])
        for g in groups:
            w.writerow([g, len(by_group[g]), picked_per_group[g], pool_per_group[g]])
        w.writerow(["TOTAL", len(data), len(test_set), len(rag_pool)])

    pool_per_repo = Counter(x["repo"] for x in rag_pool)
    n_with_overlap = 0
    with open("split_repo_overlap.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["test_url", "repo", "group", "same_repo_adrs_in_pool"])
        for x in test_set:
            n = pool_per_repo.get(x["repo"], 0)
            n_with_overlap += n > 0
            w.writerow([x["url"], x["repo"], x["group"], n])

    print(f"Seed = {SEED} | Tổng: {len(data)} ADR -> test: {len(test_set)} | RAG pool: {len(rag_pool)}")
    print(f"Số repo: toàn bộ {len({x['repo'] for x in data})} | "
          f"test {len({x['repo'] for x in test_set})} | pool {len(pool_per_repo)}")
    n_inelig = sum(not x.get("test_eligible", True) for x in data)
    if n_inelig:
        print(f"{n_inelig} ADR có Decision quá dài: chỉ nằm trong RAG pool.")
    print("\nPhân bố theo nhóm (tổng / test / pool):")
    for g in groups:
        flag = "  << n test nhỏ, chỉ nên báo cáo ở mức thăm dò" if picked_per_group[g] < 5 else ""
        print(f"  {g}: {len(by_group[g])} / {picked_per_group[g]} / {pool_per_group[g]}{flag}")
    print(f"\n{n_with_overlap}/{len(test_set)} ADR test có ADR cùng repo trong pool "
          f"-> bước 2 sẽ loại các ADR đó khi retrieve (xem split_repo_overlap.csv).")
    print("[OK] Không có URL trùng giữa test set và RAG pool.")


if __name__ == "__main__":
    main()
