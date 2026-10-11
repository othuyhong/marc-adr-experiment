"""
3b_paper_tables.py
Các bảng bổ sung dùng trong bài, tính từ kết quả của 3_evaluation_layer.py
(units.csv và evaluation_results.json). Không cần tải model, chạy vài giây.

    python 3b_paper_tables.py

Kết quả (thư mục paper_tables/):
    model_pooled.csv     mỗi model, gộp hai điều kiện: trung bình + CI 95% (bootstrap theo ADR)
    model_pairs.csv      chênh lệch ghép cặp giữa từng cặp model: CI, Wilcoxon, Holm (6 cặp)
    group_pooled.csv     mỗi nhóm chủ đề: n ADR test, F1, MCS, và chênh lệch RAG - no-RAG của F1
    length_effects.csv   tương quan Spearman giữa độ dài output và các chỉ số; tỷ lệ độ dài output/reference
    truncation_check.csv MCS của từng model, tách theo đơn vị (ADR, điều kiện) không có lượt nào bị cắt
                         và đơn vị có ít nhất một lượt bị cắt ở giới hạn token
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

SEED = 2026
N_BOOT = 5000
OUT = "paper_tables"
MODELS = ["groq-qwen3.8-27b", "groq-allam-2-7b", "groq-gpt-oss-20b", "groq-gpt-oss-120b"]
METRICS = ["bertscore_precision", "bertscore_recall", "bertscore_f1",
           "dcr_tfidf", "dcr_sbert", "mcs_tfidf", "mcs_sbert"]


def boot(values, rng):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), np.nan, np.nan
    means = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        out[i] = running
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    units = pd.read_csv("units.csv")
    models = [m for m in MODELS if m in set(units["model"])]

    # mỗi (ADR, model): trung bình hai điều kiện
    per = units.groupby(["test_id", "model"])[METRICS + ["answer_chars"]].mean()

    rows = []
    for m in models:
        x = per.xs(m, level="model")
        row = {"model": m, "n_adrs": len(x), "answer_chars": float(x["answer_chars"].mean())}
        for k in METRICS:
            row[k], row[k + "_lo"], row[k + "_hi"] = boot(x[k].values, rng)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "model_pooled.csv"), index=False)

    rows = []
    for k in ["bertscore_f1", "dcr_tfidf", "dcr_sbert", "mcs_tfidf", "mcs_sbert"]:
        block = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                d = (per[k].xs(models[i], level="model") - per[k].xs(models[j], level="model")).values
                mean, lo, hi = boot(d, rng)
                block.append({"metric": k, "model_a": models[i], "model_b": models[j],
                              "diff_a_minus_b": mean, "ci_lo": lo, "ci_hi": hi,
                              "wilcoxon_p": float(wilcoxon(d).pvalue)})
        for b, h in zip(block, holm([b["wilcoxon_p"] for b in block])):
            b["wilcoxon_p_holm"] = float(h)
        rows.extend(block)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "model_pairs.csv"), index=False)

    # theo nhóm: gộp model, mỗi ADR một giá trị
    adr = units.groupby(["test_id", "group", "condition"])[METRICS].mean().reset_index()
    rows = []
    for grp, x in adr.groupby("group"):
        pooled = x.groupby("test_id")[METRICS].mean()
        f1 = x.pivot(index="test_id", columns="condition", values="bertscore_f1")
        row = {"group": grp, "n_test_adrs": len(pooled)}
        for k in ["bertscore_f1", "mcs_tfidf", "mcs_sbert"]:
            row[k], row[k + "_lo"], row[k + "_hi"] = boot(pooled[k].values, rng)
        if {"rag", "no_rag"} <= set(f1.columns):
            row["f1_rag_minus_norag"], row["f1_diff_lo"], row["f1_diff_hi"] = boot(
                (f1["rag"] - f1["no_rag"]).values, rng)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "group_pooled.csv"), index=False)

    with open("evaluation_results.json", "r", encoding="utf-8") as f:
        rec = pd.DataFrame(json.load(f))
    rec["ref_chars"] = rec["actual_decision"].str.len()
    rows = []
    for k in ["bertscore_precision", "bertscore_recall", "bertscore_f1"]:
        rows.append({"what": f"spearman(output chars, {k}) over generations",
                     "value": float(spearmanr(rec["answer_chars"], rec[k]).statistic), "n": len(rec)})
    for k in ["mcs_tfidf", "mcs_sbert"]:
        rows.append({"what": f"spearman(mean output chars, {k}) over units",
                     "value": float(spearmanr(units["answer_chars"], units[k]).statistic), "n": len(units)})
        for m in models:
            x = units[units["model"] == m]
            rows.append({"what": f"spearman(mean output chars, {k}) within {m}",
                         "value": float(spearmanr(x["answer_chars"], x[k]).statistic), "n": len(x)})
    for m in models:
        x = rec[rec["model"] == m]
        rows.append({"what": f"median output/reference length ratio, {m}",
                     "value": float((x["answer_chars"] / x["ref_chars"]).median()), "n": len(x)})
    ref = rec.drop_duplicates("test_id")["ref_chars"]
    rows.append({"what": "median reference decision chars (test set)", "value": float(ref.median()), "n": len(ref)})
    rows.append({"what": "overall truncation rate (finish_reason = length)",
                 "value": float(rec["truncated"].mean()), "n": len(rec)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "length_effects.csv"), index=False)

    # MCS có phụ thuộc vào việc output bị cắt ở giới hạn token không?
    n_trunc = (rec.groupby(["test_id", "model", "condition"])["truncated"].sum()
               .rename("n_truncated_runs").reset_index())
    merged = units.merge(n_trunc, on=["test_id", "model", "condition"])
    rows = []
    for m in models:
        x = merged[merged["model"] == m]
        for label, part in [("no truncated run", x[x["n_truncated_runs"] == 0]),
                            ("at least one truncated run", x[x["n_truncated_runs"] > 0]),
                            ("all units", x)]:
            rows.append({"model": m, "units": label, "n_units": len(part),
                         "mcs_tfidf": float(part["mcs_tfidf"].mean()) if len(part) else np.nan,
                         "mcs_sbert": float(part["mcs_sbert"].mean()) if len(part) else np.nan,
                         "answer_chars": float(part["answer_chars"].mean()) if len(part) else np.nan})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "truncation_check.csv"), index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    for name in ["model_pooled", "model_pairs", "group_pooled", "length_effects", "truncation_check"]:
        print(f"\n=== {name}.csv ===")
        print(pd.read_csv(os.path.join(OUT, name + ".csv")).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
