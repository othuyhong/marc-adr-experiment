"""Fig. 2: BERTScore F1 and multi-run consistency by model (mean, 95% bootstrap CI over 35 ADRs).
Reads paper_tables/model_pooled.csv produced by 3b_paper_tables.py."""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "paper_tables/model_pooled.csv"
df = pd.read_csv(src)
names = {"groq-qwen3.8-27b": "Qwen3.8-27B", "groq-allam-2-7b": "ALLaM-2-7B",
         "groq-gpt-oss-20b": "GPT-OSS-20B", "groq-gpt-oss-120b": "GPT-OSS-120B"}
order = ["groq-qwen3.8-27b", "groq-allam-2-7b", "groq-gpt-oss-20b", "groq-gpt-oss-120b"]
df = df.set_index("model").loc[order]
panels = [("bertscore_f1", "BERTScore F1"), ("mcs_tfidf", r"MCS$_\mathrm{L}$ (TF-IDF)"),
          ("mcs_sbert", r"MCS$_\mathrm{S}$ (Sentence-BERT)")]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})  # embed TrueType, not Type 3
fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.9), sharey=True)
ys = list(range(len(order)))[::-1]
for ax, (key, title) in zip(axes, panels):
    m, lo, hi = df[key], df[key + "_lo"], df[key + "_hi"]
    ax.errorbar(m, ys, xerr=[m - lo, hi - m], fmt="o", color="#222222", ecolor="#222222",
                markersize=4.5, elinewidth=1.1, capsize=2.5, capthick=1.1)
    for y, v in zip(ys, m):
        ax.annotate(f"{v:.3f}", (v, y), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7, color="#222222")
    ax.set_title(title, fontsize=8.5)
    ax.set_ylim(-0.6, len(order) - 0.3)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    pad = (hi.max() - lo.min()) * 0.12
    ax.set_xlim(lo.min() - pad, hi.max() + pad)
    ax.locator_params(axis="x", nbins=4)
axes[0].set_yticks(ys)
axes[0].set_yticklabels([names[o] for o in order])
fig.tight_layout(w_pad=1.2)
fig.savefig("fig_quality_consistency.pdf")
fig.savefig("fig_quality_consistency.png", dpi=200)
