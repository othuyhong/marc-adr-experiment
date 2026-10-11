"""
3_evaluation_layer.py
BƯỚC 3 — Evaluation Layer. GỘP 3_evaluation_layer.py + 6_semantic_evaluation.py +
5_output_length_stats.py cũ vào một lần chạy.

Tính gì:
  - BERTScore P/R/F1 (generated vs. actual decision)
  - DCR, MCS, DDI theo HAI cách đo song song: TF-IDF (từ vựng) và Sentence-BERT (ngữ nghĩa)
    [R1, R2, R3#3-5]
  - Khoảng tin cậy 95% bằng bootstrap THEO ADR (đơn vị lấy mẫu lại là ADR, không phải
    từng lượt sinh, vì các lượt sinh của cùng một ADR không độc lập)        [R3#2, #15]
  - Kiểm định ghép cặp RAG vs no-RAG (Wilcoxon signed-rank theo ADR + CI của chênh lệch,
    hiệu chỉnh Holm trên 4 model)                                          [R3#15]
  - Số ADR test theo từng nhóm, kèm theo mọi bảng theo nhóm                [R3#9, #16]
  - Mức đồng thuận giữa hai cách đo (Spearman) và độ ổn định thứ hạng model
    (chất lượng vs. nhất quán) qua bootstrap                               [R3#8]
  - Độ dài output, tỷ lệ bị cắt (finish_reason thật), token suy luận       [R3#17]

Yêu cầu:
    pip install bert-score sentence-transformers scikit-learn pandas scipy

Cách chạy:
    python 3_evaluation_layer.py
    python 3_evaluation_layer.py --in generation_T0.json --outdir eval_T0     (thí nghiệm temperature)

Kết quả (trong --outdir, mặc định là thư mục hiện tại):
    evaluation_results.json          mỗi lượt sinh + metric
    units.csv                        mỗi (ADR, model, điều kiện): trung bình các lượt + MCS
    table_by_model.csv               mọi metric theo model x điều kiện, kèm CI 95%
    table_rag_paired.csv             RAG - no-RAG theo model: chênh lệch, CI, p, p Holm
    table_ddi.csv                    DDI theo điều kiện và theo nhóm, kèm CI
    table_ddi_paired.csv             DDI: RAG - no-RAG ghép cặp theo ADR (cả 4 model, và khi bỏ từng model)
    table_by_group.csv               metric theo nhóm x điều kiện, kèm n ADR test
    table_metric_agreement.csv       TF-IDF vs SBERT; độ dài output vs MCS
    table_rank_stability.csv         thứ hạng model và độ ổn định qua bootstrap
    table_output_length.csv          độ dài, tỷ lệ bị cắt, token suy luận
    test_group_counts.csv            số ADR test theo nhóm
"""
import argparse
import hashlib
import json
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

N_BOOT = 5000
SEED = 42
SBERT_MODEL = "all-MiniLM-L6-v2"
SBERT_MAX_SEQ = 512        # mặc định của model là 256 word-piece; nâng lên để ít cắt output dài
MODELS = ["groq-gpt-oss-120b", "groq-gpt-oss-20b", "groq-qwen3.8-27b", "groq-allam-2-7b"]

RECORD_METRICS = ["bertscore_precision", "bertscore_recall", "bertscore_f1",
                  "dcr_tfidf", "dcr_sbert"]
UNIT_METRICS = RECORD_METRICS + ["mcs_tfidf", "mcs_sbert"]


# ----------------------------------------------------------------------
# THỐNG KÊ
# ----------------------------------------------------------------------
def boot_ci(values, rng):
    """Trung bình và CI 95% (percentile bootstrap). values: mỗi phần tử là một ADR."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0
    if len(v) == 1:
        return float(v[0]), np.nan, np.nan, 1
    means = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi), len(v)


def holm(pvals):
    """p hiệu chỉnh Holm; NaN được giữ nguyên."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    idx = [i for i in np.argsort(p) if not np.isnan(p[i])]
    m, running = len(idx), 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def safe_wilcoxon(diffs):
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 6 or np.allclose(d, 0):
        return np.nan
    try:
        return float(wilcoxon(d).pvalue)
    except ValueError:
        return np.nan


def safe_spearman(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3 or np.ptp(a[m]) == 0 or np.ptp(b[m]) == 0:
        return np.nan, int(m.sum())
    return float(spearmanr(a[m], b[m]).statistic), int(m.sum())


def mean_pairwise(sim_matrix):
    n = sim_matrix.shape[0]
    return float(np.mean([sim_matrix[i, j] for i, j in combinations(range(n), 2)]))


# ----------------------------------------------------------------------
# ĐO ĐỘ TƯƠNG ĐỒNG
# ----------------------------------------------------------------------
def compute_bertscore(cands, refs):
    from bert_score import score
    print("  (lần đầu sẽ tải roberta-large ~1.4GB)")
    P, R, F1 = score(cands, refs, lang="en", verbose=True)
    return [float(x) for x in P], [float(x) for x in R], [float(x) for x in F1]


def sbert_embed(texts):
    """Trả về (ma trận embedding đã chuẩn hóa, tỷ lệ văn bản dài hơn giới hạn của encoder)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SBERT_MODEL)
    model.max_seq_length = SBERT_MAX_SEQ
    emb = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    try:
        lens = [len(ids) for ids in model.tokenizer(texts, truncation=False)["input_ids"]]
        over = float(np.mean([n > SBERT_MAX_SEQ for n in lens]))
    except Exception:
        over = np.nan
    return np.asarray(emb), over


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="generation_results.json")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    out = lambda name: os.path.join(args.outdir, name)
    rng = np.random.default_rng(SEED)

    with open(args.infile, "r", encoding="utf-8") as f:
        results = [r for r in json.load(f) if r["model"] in MODELS]
    if not results:
        raise SystemExit("Không có bản ghi nào của 4 model chính thức.")
    cfgs = {r.get("config_id") for r in results}
    if len(cfgs) > 1:
        raise SystemExit(f"{args.infile} trộn nhiều cấu hình sinh ({cfgs}) - không đánh giá chung được.")

    models = [m for m in MODELS if any(r["model"] == m for r in results)]
    conditions = sorted({r["condition"] for r in results})
    n_rec = len(results)
    print(f"{n_rec} lượt sinh | {len({r['test_id'] for r in results})} ADR | "
          f"{len(models)} model | điều kiện: {conditions}")

    # ---- 1. BERTScore ----
    print("\n[1/5] BERTScore...")
    # Dùng lại BERTScore đã tính ở lần chạy trước (nếu cặp output/reference không đổi)
    pair_key = lambda r: hashlib.md5((r["generated_decision"] + "\x00" + r["actual_decision"])
                                     .encode("utf-8")).hexdigest()
    cache = {}
    if os.path.exists(out("evaluation_results.json")):
        try:
            with open(out("evaluation_results.json"), "r", encoding="utf-8") as f:
                for old in json.load(f):
                    if "bertscore_f1" in old:
                        cache[pair_key(old)] = (old["bertscore_precision"], old["bertscore_recall"],
                                                old["bertscore_f1"])
        except (ValueError, KeyError):
            cache = {}
    todo = [i for i, r in enumerate(results) if pair_key(r) not in cache]
    print(f"  dùng lại {n_rec - len(todo)} điểm đã tính, cần tính mới {len(todo)}")
    if todo:
        P, R, F1 = compute_bertscore([results[i]["generated_decision"] for i in todo],
                                     [results[i]["actual_decision"] for i in todo])
        for j, i in enumerate(todo):
            cache[pair_key(results[i])] = (P[j], R[j], F1[j])
    for r in results:
        r["bertscore_precision"], r["bertscore_recall"], r["bertscore_f1"] = cache[pair_key(r)]

    # ---- 2. Vector hóa: TF-IDF và Sentence-BERT ----
    print("\n[2/5] TF-IDF và Sentence-BERT...")
    decisions = [r["generated_decision"] for r in results]
    contexts = [r["context"] for r in results]

    # DCR: một không gian TF-IDF chung cho context + decision (như bản cũ)
    tf_joint = TfidfVectorizer(stop_words="english", max_features=5000).fit(contexts + decisions)
    ctx_tf, dec_tf_joint = tf_joint.transform(contexts), tf_joint.transform(decisions)
    # MCS/DDI: không gian TF-IDF chỉ gồm các decision sinh ra (như bản cũ)
    dec_tf = TfidfVectorizer(stop_words="english", max_features=5000).fit_transform(decisions)

    uniq = sorted(set(decisions) | set(contexts))
    pos = {t: i for i, t in enumerate(uniq)}
    emb, frac_over = sbert_embed(uniq)
    dec_sb = emb[[pos[t] for t in decisions]]
    ctx_sb = emb[[pos[t] for t in contexts]]
    if not np.isnan(frac_over):
        print(f"  {frac_over:.1%} văn bản dài hơn {SBERT_MAX_SEQ} word-piece (bị encoder cắt)")

    for i, r in enumerate(results):
        r["dcr_tfidf"] = float(cosine_similarity(ctx_tf[i], dec_tf_joint[i])[0, 0])
        r["dcr_sbert"] = float(ctx_sb[i] @ dec_sb[i])
        r["answer_chars"] = len(r["generated_decision"])

    # ---- 3. MCS: giữa các lượt lặp của cùng (ADR, model, điều kiện) ----
    print("\n[3/5] MCS...")
    by_unit = defaultdict(list)
    for i, r in enumerate(results):
        by_unit[(r["test_id"], r["model"], r["condition"])].append(i)
    mcs = {}
    for key, idx in by_unit.items():
        if len(idx) < 2:
            continue
        mcs[key] = (mean_pairwise(cosine_similarity(dec_tf[idx])),
                    mean_pairwise(dec_sb[idx] @ dec_sb[idx].T))
        for i in idx:
            results[i]["mcs_tfidf"], results[i]["mcs_sbert"] = mcs[key]

    # ---- 4. DDI: giữa các model cho cùng (ADR, điều kiện, lượt) ----
    print("\n[4/5] DDI...")
    by_ddi = defaultdict(list)
    for i, r in enumerate(results):
        by_ddi[(r["test_id"], r["condition"], r["run_index"])].append(i)
    ddi_rows, ddi_subset_rows, n_ddi_incomplete = [], [], 0
    for (tid, cond, run), idx in by_ddi.items():
        if {results[i]["model"] for i in idx} != set(models) or len(idx) != len(models):
            n_ddi_incomplete += 1          # chỉ tính khi đủ mọi model, để các ô so sánh được
            continue
        if len(models) >= 3:               # DDI khi bỏ từng model: tác động có do một model gây ra không
            for drop in models:
                sub = [i for i in idx if results[i]["model"] != drop]
                ddi_subset_rows.append({
                    "test_id": tid, "condition": cond, "run_index": run, "excluded_model": drop,
                    "ddi_tfidf": 1 - mean_pairwise(cosine_similarity(dec_tf[sub])),
                    "ddi_sbert": 1 - mean_pairwise(dec_sb[sub] @ dec_sb[sub].T),
                })
        ddi_rows.append({
            "test_id": tid, "condition": cond, "run_index": run, "group": results[idx[0]]["group"],
            "ddi_tfidf": 1 - mean_pairwise(cosine_similarity(dec_tf[idx])),
            "ddi_sbert": 1 - mean_pairwise(dec_sb[idx] @ dec_sb[idx].T),
        })
    ddi_df = pd.DataFrame(ddi_rows)

    with open(out("evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 5. TỔNG HỢP
    # ------------------------------------------------------------------
    print("\n[5/5] Tổng hợp và thống kê...")
    df = pd.DataFrame(results)
    for col in ["mcs_tfidf", "mcs_sbert", "completion_tokens", "reasoning_tokens",
                "truncated", "had_think_block"]:
        if col not in df.columns:
            df[col] = np.nan

    # Đơn vị phân tích: (ADR, model, điều kiện) = trung bình các lượt lặp
    units = (df.groupby(["test_id", "group", "model", "condition"], as_index=False)
               .agg(n_runs=("run_index", "nunique"),
                    answer_chars=("answer_chars", "mean"),
                    **{m: (m, "mean") for m in UNIT_METRICS}))
    units.to_csv(out("units.csv"), index=False)

    n_runs_expected = int(units["n_runs"].max())
    incomplete = units[units["n_runs"] < n_runs_expected]
    n_adr = units["test_id"].nunique()
    expected_units = n_adr * len(models) * len(conditions)
    if len(incomplete) or len(units) < expected_units:
        print(f"  [CẢNH BÁO] Dữ liệu CHƯA ĐỦ: {len(units)}/{expected_units} đơn vị, "
              f"{len(incomplete)} đơn vị có ít hơn {n_runs_expected} lượt. "
              f"Chạy tiếp 2_generation_layer.py trước khi đưa số vào bài.")
    if n_ddi_incomplete:
        print(f"  [CẢNH BÁO] {n_ddi_incomplete} ô DDI thiếu model, đã bỏ khỏi DDI.")

    # ---- Số ADR test theo nhóm ----
    grp_n = units.groupby("group")["test_id"].nunique().rename("n_test_adrs")
    grp_n.to_csv(out("test_group_counts.csv"))

    # ---- Bảng theo model x điều kiện ----
    rows = []
    for (model, cond), g in units.groupby(["model", "condition"]):
        row = {"model": model, "condition": cond, "n_adrs": g["test_id"].nunique()}
        for m in UNIT_METRICS:
            mean, lo, hi, _ = boot_ci(g[m].values, rng)
            row[m], row[m + "_lo"], row[m + "_hi"] = mean, lo, hi
        rows.append(row)
    by_model = pd.DataFrame(rows)
    by_model.to_csv(out("table_by_model.csv"), index=False)

    # ---- RAG vs no-RAG, ghép cặp theo ADR ----
    paired = pd.DataFrame()
    if {"rag", "no_rag"} <= set(conditions):
        rows = []
        for m in UNIT_METRICS:
            wide = units.pivot_table(index=["test_id", "model"], columns="condition", values=m)
            d = (wide["rag"] - wide["no_rag"]).dropna().rename("diff").reset_index()
            block = []
            for model in models + ["ALL_MODELS"]:
                # ALL_MODELS: trung bình chênh lệch của các model trong từng ADR -> 1 giá trị/ADR
                v = (d.groupby("test_id")["diff"].mean() if model == "ALL_MODELS"
                     else d[d["model"] == model].set_index("test_id")["diff"]).values
                mean, lo, hi, n = boot_ci(v, rng)
                block.append({"metric": m, "model": model, "n_adrs": n, "diff_rag_minus_norag": mean,
                              "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": safe_wilcoxon(v)})
            adj = holm([b["wilcoxon_p"] for b in block[:-1]])
            for b, a in zip(block[:-1], adj):
                b["wilcoxon_p_holm"] = a
            block[-1]["wilcoxon_p_holm"] = np.nan
            rows.extend(block)
        paired = pd.DataFrame(rows)
        paired.to_csv(out("table_rag_paired.csv"), index=False)

    # ---- DDI ----
    ddi_table = pd.DataFrame()
    if len(ddi_df):
        ddi_adr = ddi_df.groupby(["test_id", "group", "condition"], as_index=False)[
            ["ddi_tfidf", "ddi_sbert"]].mean()
        rows = []
        for keys, label in [(["condition"], "ALL")] + [(["condition", "group"], None)]:
            for k, g in ddi_adr.groupby(keys):
                k = k if isinstance(k, tuple) else (k,)
                row = {"condition": k[0], "group": label or k[1], "n_adrs": g["test_id"].nunique()}
                for m in ["ddi_tfidf", "ddi_sbert"]:
                    mean, lo, hi, _ = boot_ci(g[m].values, rng)
                    row[m], row[m + "_lo"], row[m + "_hi"] = mean, lo, hi
                rows.append(row)
        ddi_table = pd.DataFrame(rows)
        ddi_table.to_csv(out("table_ddi.csv"), index=False)

    # ---- DDI: RAG - no-RAG, ghép cặp theo ADR ----
    ddi_paired = pd.DataFrame()
    if len(ddi_df) and {"rag", "no_rag"} <= set(conditions):
        sets = [("ALL_MODELS", ddi_df)]
        if ddi_subset_rows:
            sub_df = pd.DataFrame(ddi_subset_rows)
            sets += [(f"excluding {m}", sub_df[sub_df["excluded_model"] == m]) for m in models]
        rows = []
        rng_ddi = np.random.default_rng(SEED + 1)   # bộ sinh số riêng: các bảng khác giữ nguyên số như lần chạy trước
        for label, frame in sets:
            per_adr = frame.groupby(["test_id", "condition"])[["ddi_tfidf", "ddi_sbert"]].mean()
            for m in ["ddi_tfidf", "ddi_sbert"]:
                wide = per_adr[m].unstack("condition").dropna()
                d = (wide["rag"] - wide["no_rag"]).values
                mean, lo, hi, n = boot_ci(d, rng_ddi)
                rows.append({"metric": m, "models": label, "n_adrs": n,
                             "no_rag": float(wide["no_rag"].mean()), "rag": float(wide["rag"].mean()),
                             "diff_rag_minus_norag": mean, "ci_lo": lo, "ci_hi": hi,
                             "wilcoxon_p": safe_wilcoxon(d),
                             "share_adrs_increase": float(np.mean(d > 0))})
        ddi_paired = pd.DataFrame(rows)
        ddi_paired.to_csv(out("table_ddi_paired.csv"), index=False)

    # ---- Theo nhóm x điều kiện (gộp model: 1 giá trị/ADR) ----
    adr_cond = units.groupby(["test_id", "group", "condition"], as_index=False)[UNIT_METRICS].mean()
    rows = []
    for (grp, cond), g in adr_cond.groupby(["group", "condition"]):
        row = {"group": grp, "condition": cond, "n_test_adrs": g["test_id"].nunique()}
        for m in ["bertscore_f1", "dcr_tfidf", "dcr_sbert", "mcs_tfidf", "mcs_sbert"]:
            mean, lo, hi, _ = boot_ci(g[m].values, rng)
            row[m], row[m + "_lo"], row[m + "_hi"] = mean, lo, hi
        rows.append(row)
    pd.DataFrame(rows).to_csv(out("table_by_group.csv"), index=False)

    # ---- Hai cách đo có đồng thuận không; độ dài có giải thích MCS không ----
    rows = []
    pairs = [("MCS: tfidf vs sbert", units, "mcs_tfidf", "mcs_sbert"),
             ("DCR: tfidf vs sbert", df, "dcr_tfidf", "dcr_sbert"),
             ("độ dài output vs MCS tfidf", units, "answer_chars", "mcs_tfidf"),
             ("độ dài output vs MCS sbert", units, "answer_chars", "mcs_sbert")]
    if len(ddi_df):
        pairs.append(("DDI: tfidf vs sbert", ddi_df, "ddi_tfidf", "ddi_sbert"))
    for label, frame, a, b in pairs:
        rho, n = safe_spearman(frame[a].values.astype(float), frame[b].values.astype(float))
        rows.append({"comparison": label, "spearman_rho": rho, "n": n})
    agreement = pd.DataFrame(rows)
    agreement.to_csv(out("table_metric_agreement.csv"), index=False)

    # ---- Thứ hạng model: chất lượng vs nhất quán, và độ ổn định qua bootstrap ----
    rows = []
    for cond in conditions:
        u = units[units["condition"] == cond]
        mats = {m: u.pivot_table(index="test_id", columns="model", values=m).reindex(columns=models)
                for m in ["bertscore_f1", "mcs_tfidf", "mcs_sbert"]}
        keep = mats["bertscore_f1"].dropna().index
        for m in mats:
            keep = keep.intersection(mats[m].dropna().index)
        if len(keep) < 2 or len(models) < 2:
            continue
        arr = {m: mats[m].loc[keep].values for m in mats}
        idx = rng.integers(0, len(keep), size=(N_BOOT, len(keep)))
        boot = {m: arr[m][idx].mean(axis=1) for m in arr}          # (N_BOOT, n_model)
        order = {m: [models[i] for i in np.argsort(-arr[m].mean(axis=0))] for m in arr}
        for m in arr:
            top_obs = int(np.argmax(arr[m].mean(axis=0)))
            rows.append({"condition": cond, "what": f"thứ hạng theo {m}",
                         "value": " > ".join(order[m]),
                         "bootstrap_share": float(np.mean(boot[m].argmax(axis=1) == top_obs)),
                         "note": "tỷ lệ bootstrap mà model đứng đầu giữ nguyên"})
        for mc in ["mcs_tfidf", "mcs_sbert"]:
            tau = kendalltau(arr["bertscore_f1"].mean(axis=0), arr[mc].mean(axis=0)).statistic
            differ = np.mean(boot["bertscore_f1"].argmax(axis=1) != boot[mc].argmax(axis=1))
            rows.append({"condition": cond, "what": f"bertscore_f1 vs {mc}",
                         "value": f"kendall_tau={tau:.3f} ({len(models)} model)",
                         "bootstrap_share": float(differ),
                         "note": "tỷ lệ bootstrap mà model tốt nhất theo F1 KHÁC model nhất quán nhất"})
        tau = kendalltau(arr["mcs_tfidf"].mean(axis=0), arr["mcs_sbert"].mean(axis=0)).statistic
        same = np.mean(boot["mcs_tfidf"].argmax(axis=1) == boot["mcs_sbert"].argmax(axis=1))
        rows.append({"condition": cond, "what": "mcs_tfidf vs mcs_sbert",
                     "value": f"kendall_tau={tau:.3f} ({len(models)} model)",
                     "bootstrap_share": float(same),
                     "note": "tỷ lệ bootstrap mà hai cách đo chọn CÙNG model nhất quán nhất"})
    rank = pd.DataFrame(rows)
    rank.to_csv(out("table_rank_stability.csv"), index=False)

    # ---- Độ dài output / bị cắt / token suy luận ----
    num = lambda s: pd.to_numeric(s, errors="coerce")
    length = (df.assign(completion_tokens=num(df["completion_tokens"]),
                        reasoning_tokens=num(df["reasoning_tokens"]),
                        truncated=num(df["truncated"]), had_think_block=num(df["had_think_block"]))
                .groupby(["model", "condition"])
                .agg(n=("answer_chars", "size"),
                     answer_chars_mean=("answer_chars", "mean"),
                     answer_chars_median=("answer_chars", "median"),
                     completion_tokens_mean=("completion_tokens", "mean"),
                     reasoning_tokens_mean=("reasoning_tokens", "mean"),
                     truncation_rate=("truncated", "mean"),
                     think_block_rate=("had_think_block", "mean"))
                .reset_index())
    length.to_csv(out("table_output_length.csv"), index=False)

    # ------------------------------------------------------------------
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    fmt = lambda t: t.round(3).to_string(index=False)
    print(f"\n=== Số ADR test theo nhóm ===\n{grp_n.to_string()}")
    print("\n=== Theo model x điều kiện (trung bình; CI 95% trong table_by_model.csv) ===")
    print(fmt(by_model[["model", "condition", "n_adrs"] + UNIT_METRICS]))
    if len(paired):
        print("\n=== RAG - no-RAG (ghép cặp theo ADR) ===")
        print(fmt(paired[paired["metric"].isin(["bertscore_f1", "mcs_tfidf", "mcs_sbert"])]))
    if len(ddi_table):
        print("\n=== DDI theo điều kiện ===")
        print(fmt(ddi_table[ddi_table["group"] == "ALL"]))
    if len(ddi_paired):
        print("\n=== DDI: RAG - no-RAG (ghép cặp theo ADR; 'excluding X' = tính trên 3 model còn lại) ===")
        print(fmt(ddi_paired))
    print(f"\n=== Đồng thuận giữa các cách đo ===\n{fmt(agreement)}")
    if len(rank):
        print(f"\n=== Thứ hạng model ===\n{rank.round(3).to_string(index=False)}")
    print(f"\n=== Độ dài output ===\n{fmt(length)}")
    print(f"\nĐã lưu các bảng vào: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
