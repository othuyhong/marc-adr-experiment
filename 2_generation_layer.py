"""
2_generation_layer.py
BƯỚC 2 — Generation Layer: 4 model trên Groq sinh Decision từ Context,
2 điều kiện (no_rag / rag), lặp N_REPEATS lần mỗi tổ hợp.

THAY ĐỔI SO VỚI BẢN CŨ:
  [R3#11] Retrieval LOẠI mọi ADR cùng repo với ADR đang test; lưu danh sách ADR đã
          retrieve (url, repo, độ tương đồng) vào từng bản ghi rag.
  [R3#17] Cả 4 model dùng CÙNG một giới hạn MAX_COMPLETION_TOKENS (bản cũ: 300 vs 700).
          Lưu finish_reason và usage thật từ API (không còn ước lượng 4 ký tự/token).
  [R3#17] Tách khối <think>...</think> khỏi output trước khi lưu; bản gốc giữ ở raw_output.
  Output rỗng sau khi tách: thử lại, nếu vẫn rỗng thì ghi vào generation_failures.json
          (không lưu vào kết quả, nên không cần 2b_cleanup_empty_results.py nữa).
  Checkpoint gắn với cấu hình: nếu file kết quả cũ được sinh bằng split/temperature/
          max_tokens khác thì script DỪNG thay vì âm thầm trộn hai loại kết quả.
  429: đợi đúng theo header retry-after; nếu phải đợi quá lâu (hết hạn mức ngày) thì
          tạm bỏ model đó trong lần chạy này, các model khác vẫn chạy tiếp.

Yêu cầu:
    pip install requests scikit-learn
    $env:GROQ_API_KEY="..."        (PowerShell)

Cách chạy:
    python 2_generation_layer.py

Thí nghiệm temperature (R3#18), ghi ra file riêng để không trộn với kết quả chính:
    python 2_generation_layer.py --temperature 0   --conditions no_rag --limit 10 --out generation_T0.json
    python 2_generation_layer.py --temperature 0.7 --conditions no_rag --limit 10 --out generation_T07.json

Bị dừng giữa chừng -> chạy lại đúng lệnh cũ, script tự bỏ qua phần đã xong.
"""
import argparse
import hashlib
import json
import os
import random
import re
import time

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# CẤU HÌNH
# ----------------------------------------------------------------------
N_REPEATS = 5
TOP_K_RAG = 4
TEMPERATURE = 0.3
MAX_COMPLETION_TOKENS = 1024     # DÙNG CHUNG cho cả 4 model
MIN_ANSWER_CHARS = 10
MAX_ATTEMPTS = 3                 # số lần thử cho mỗi tổ hợp trong một lần chạy
LONG_WAIT_SECONDS = 600          # retry-after dài hơn mức này = coi như hết hạn mức ngày
SECONDS_BETWEEN_CALLS = 1.0
SAMPLE_SEED = 42                 # dùng cho --limit

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# tên dùng trong bài -> tên model trên Groq
MODELS = {
    "groq-gpt-oss-120b": "openai/gpt-oss-120b",
    "groq-gpt-oss-20b": "openai/gpt-oss-20b",
    "groq-qwen3.8-27b": "qwen/qwen3.8-27b",      # thay qwen3.6-27b (Groq ngừng 14/9/2026)
    "groq-allam-2-7b": "allam-2-7b",             # thay llama-3.1-8b-instant (Groq ngừng 16/8/2026)
}
# Tham số riêng của từng model. Mục tiêu: phần suy luận KHÔNG ăn vào MAX_COMPLETION_TOKENS.
#   - GPT-OSS: mức thấp nhất là "low" (chạy thử: chỉ 7-14 token suy luận mỗi lượt).
#   - Qwen3.8: "none" = chế độ instruct, không suy luận. Chạy thử với "low" thì 44% token
#     output là suy luận (TB 308 token/lượt), làm câu trả lời ngắn đi và không so được với model khác.
# Số token suy luận thật của từng lượt vẫn được lưu để báo cáo.
# Lưu ý: allam-2-7b chỉ có cửa sổ ngữ cảnh 4.096 token (prompt + output).
EXTRA_PARAMS = {
    "groq-gpt-oss-120b": {"reasoning_effort": "low"},
    "groq-gpt-oss-20b": {"reasoning_effort": "low"},
    "groq-qwen3.8-27b": {"reasoning_effort": "none"},
    "groq-allam-2-7b": {},
}
ALL_CONDITIONS = ["no_rag", "rag"]

SYSTEM_PROMPT = ("This is an Architectural Decision Record for a software. "
                 "Give a ## Decision corresponding to the ## Context provided.")

FAILURES_FILE = "generation_failures.json"


# ----------------------------------------------------------------------
# GỌI API
# ----------------------------------------------------------------------
def call_groq(model, prompt, temperature):
    body = {
        "model": MODELS[model],
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }
    body.update(EXTRA_PARAMS.get(model, {}))
    r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                      json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    return {
        "raw": msg.get("content") or "",
        "reasoning": msg.get("reasoning") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage") or {},
    }


THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_OPEN = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)


def strip_think(text):
    """Bỏ khối suy luận <think>. Khối chưa đóng (bị cắt giữa lúc suy luận) -> bỏ đến hết."""
    cleaned = THINK_CLOSED.sub("", text)
    cleaned = THINK_OPEN.sub("", cleaned)
    cleaned = cleaned.strip()
    return cleaned, cleaned != text.strip()


def retry_after_seconds(response):
    try:
        return float(response.headers.get("retry-after", ""))
    except (TypeError, ValueError):
        return 30.0


# ----------------------------------------------------------------------
# RAG RETRIEVAL — TF-IDF, loại ADR cùng repo
# ----------------------------------------------------------------------
def build_rag_index(rag_pool):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    matrix = vectorizer.fit_transform([x["context"] for x in rag_pool])
    return vectorizer, matrix


def retrieve_top_k(item, vectorizer, matrix, rag_pool, k=TOP_K_RAG):
    """Trả về (danh sách ADR, điểm tương đồng, số ADR cùng repo đã bị loại)."""
    sims = cosine_similarity(vectorizer.transform([item["context"]]), matrix)[0]
    allowed = [i for i, x in enumerate(rag_pool)
               if x["repo"] != item["repo"] and x["url"] != item["url"]]
    n_excluded = len(rag_pool) - len(allowed)
    top = sorted(allowed, key=lambda i: sims[i], reverse=True)[:k]
    return [rag_pool[i] for i in top], [float(sims[i]) for i in top], n_excluded


def build_prompt(context, retrieved=None):
    if retrieved:
        block = "\n\n".join(
            f"## Related Example\nContext: {ex['context'][:400]}\nDecision: {ex['decision'][:300]}"
            for ex in retrieved)
        return f"{block}\n\n## Context\n{context}\n\n## Decision\n"
    return f"## Context\n{context}\n\n## Decision\n"


# ----------------------------------------------------------------------
# CHECKPOINT
# ----------------------------------------------------------------------
def make_test_id(item):
    return hashlib.md5(item["url"].encode("utf-8")).hexdigest()[:10]


def combo_key(test_id, model, condition, run_index):
    return f"{test_id}|{model}|{condition}|{run_index}"


def config_id(test_set, rag_pool, temperature):
    """Dấu vân tay của mọi thứ ảnh hưởng đến output. Khác dấu = không được trộn kết quả."""
    cfg = {
        "split": hashlib.md5("\n".join(
            sorted(x["url"] for x in test_set) + ["--"] + sorted(x["url"] for x in rag_pool)
        ).encode("utf-8")).hexdigest(),
        "temperature": temperature,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "top_k": TOP_K_RAG,
        "models": MODELS,
        "extra": EXTRA_PARAMS,
        "system": SYSTEM_PROMPT,
        "retrieval": "tfidf-exclude-same-repo",
    }
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, obj):
    """Ghi qua file tạm rồi đổi tên -> mất điện giữa lúc ghi không làm hỏng file kết quả."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS, choices=ALL_CONDITIONS)
    ap.add_argument("--limit", type=int, default=None,
                    help="chỉ chạy trên N ADR test lấy ngẫu nhiên (seed cố định)")
    ap.add_argument("--out", default="generation_results.json")
    args = ap.parse_args()

    if not GROQ_API_KEY:
        raise SystemExit("Chưa set GROQ_API_KEY.")

    test_set = load_json("gold_test_set.json", None)
    rag_pool = load_json("rag_pool.json", None)
    if test_set is None or rag_pool is None:
        raise SystemExit("Thiếu gold_test_set.json / rag_pool.json. Chạy 1_4_split_dataset.py trước.")

    cfg = config_id(test_set, rag_pool, args.temperature)
    results = load_json(args.out, [])
    old_cfgs = {r.get("config_id") for r in results}
    if results and old_cfgs != {cfg}:
        raise SystemExit(
            f"{args.out} chứa kết quả sinh bằng cấu hình khác ({sorted(map(str, old_cfgs))}) "
            f"so với cấu hình hiện tại ({cfg}): dataset, temperature hoặc max_tokens đã đổi.\n"
            f"Đổi tên/xóa file cũ, hoặc dùng --out để ghi sang file khác.")

    run_items = test_set
    if args.limit is not None and args.limit < len(test_set):
        run_items = random.Random(SAMPLE_SEED).sample(test_set, args.limit)

    vectorizer, matrix = build_rag_index(rag_pool)
    done = {combo_key(r["test_id"], r["model"], r["condition"], r["run_index"]) for r in results}
    total = len(run_items) * len(MODELS) * len(args.conditions) * N_REPEATS
    print(f"Cấu hình {cfg} | T={args.temperature} | max_completion_tokens={MAX_COMPLETION_TOKENS}")
    print(f"Test: {len(run_items)} ADR | RAG pool: {len(rag_pool)} | "
          f"cần {total} lượt gọi, đã có {len(done)}")

    failures = load_json(FAILURES_FILE, [])
    blocked = {}          # model -> lý do tạm bỏ trong lần chạy này
    n_new = 0

    for item in run_items:
        test_id = make_test_id(item)
        retrieved, sims, n_excluded = retrieve_top_k(item, vectorizer, matrix, rag_pool)

        for condition in args.conditions:
            use_rag = condition == "rag"
            prompt = build_prompt(item["context"], retrieved if use_rag else None)

            for model in MODELS:
                for run_index in range(1, N_REPEATS + 1):
                    key = combo_key(test_id, model, condition, run_index)
                    if key in done or model in blocked:
                        continue

                    last_error = None
                    saved = False
                    for attempt in range(1, MAX_ATTEMPTS + 1):
                        try:
                            out = call_groq(model, prompt, args.temperature)
                        except requests.exceptions.HTTPError as e:
                            status = e.response.status_code if e.response is not None else 0
                            last_error = f"HTTP {status}"
                            if status == 429:
                                wait = retry_after_seconds(e.response)
                                if wait > LONG_WAIT_SECONDS:
                                    blocked[model] = f"hết hạn mức, cần đợi {wait / 60:.0f} phút"
                                    break
                                print(f"    [429] {model}: đợi {wait:.0f}s")
                                time.sleep(wait + 1)
                            elif status >= 500:
                                time.sleep(10)
                            else:
                                body = e.response.text[:200] if e.response is not None else ""
                                if status in (401, 403, 404):
                                    # sai key hoặc model không tồn tại: mọi lượt sau cũng sẽ lỗi
                                    blocked[model] = f"HTTP {status}: {body}"
                                else:
                                    # lỗi riêng của lượt này (vd prompt dài quá cửa sổ ngữ cảnh)
                                    last_error = f"HTTP {status}: {body}"
                                break
                            continue
                        except requests.exceptions.RequestException as e:
                            last_error = f"mạng: {e}"
                            time.sleep(10)
                            continue

                        answer, had_think = strip_think(out["raw"])
                        if len(answer) < MIN_ANSWER_CHARS:
                            last_error = (f"output rỗng sau khi bỏ <think> "
                                          f"(finish_reason={out['finish_reason']})")
                            time.sleep(SECONDS_BETWEEN_CALLS)
                            continue

                        usage = out["usage"]
                        details = usage.get("completion_tokens_details") or {}
                        rec = {
                            "test_id": test_id,
                            "url": item["url"],
                            "repo": item["repo"],
                            "group": item.get("group", "unknown"),
                            "model": model,
                            "condition": condition,
                            "run_index": run_index,
                            "context": item["context"],
                            "actual_decision": item["decision"],
                            "generated_decision": answer,
                            "finish_reason": out["finish_reason"],
                            "truncated": out["finish_reason"] == "length",
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"),
                            "reasoning_tokens": details.get("reasoning_tokens"),
                            "reasoning_chars": len(out["reasoning"]),
                            "had_think_block": had_think,
                            "temperature": args.temperature,
                            "max_completion_tokens": MAX_COMPLETION_TOKENS,
                            "config_id": cfg,
                            "timestamp": time.time(),
                        }
                        if had_think:
                            rec["raw_output"] = out["raw"]
                        if use_rag:
                            rec["retrieved"] = [
                                {"url": ex["url"], "repo": ex["repo"], "similarity": round(s, 4)}
                                for ex, s in zip(retrieved, sims)]
                            rec["n_same_repo_excluded"] = n_excluded
                        results.append(rec)
                        saved = True
                        done.add(key)
                        n_new += 1
                        save_json(args.out, results)
                        flag = " [BỊ CẮT]" if rec["truncated"] else ""
                        print(f"[OK] {test_id} | {model} | {condition} | run {run_index}{flag}")
                        time.sleep(SECONDS_BETWEEN_CALLS)
                        break
                    if not saved and model not in blocked:
                        failures.append({"key": key, "config_id": cfg, "error": last_error,
                                         "timestamp": time.time()})
                        save_json(FAILURES_FILE, failures)
                        print(f"[BỎ QUA] {key}: {last_error}")

                    if model in blocked:
                        print(f"[TẠM BỎ {model}] {blocked[model]}")

    n_done = sum(1 for r in results
                 if combo_key(r["test_id"], r["model"], r["condition"], r["run_index"]) in done)
    print("\n=== KẾT THÚC LẦN CHẠY ===")
    print(f"Sinh mới: {n_new} | Tổng trong {args.out}: {n_done}")
    for m, why in blocked.items():
        print(f"  Model tạm bỏ: {m} ({why})")
    remaining = total - sum(
        1 for it in run_items for c in args.conditions for m in MODELS
        for i in range(1, N_REPEATS + 1) if combo_key(make_test_id(it), m, c, i) in done)
    if remaining:
        print(f"Còn thiếu {remaining}/{total} lượt -> chạy lại đúng lệnh này để làm tiếp.")
    else:
        print("ĐÃ ĐỦ toàn bộ tổ hợp.")


if __name__ == "__main__":
    main()
