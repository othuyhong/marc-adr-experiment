# MARC: Generating Architectural Decision Records with Multi-Run Consistency Evaluation

Code, data and results for the paper *"MARC: A Four-Layer Framework for Generating
Architectural Decision Records in Microservices with Multi-Run Consistency Evaluation"*
(EAI FISAT 2026).

The study asks how consistent LLM-generated ADR decisions are when the same context is
submitted repeatedly. Four open-weight models each generate the *Decision* of 35 held-out
ADRs five times, with and without retrieval augmentation (1,400 generations), and the
outputs are scored with BERTScore and three similarity-based metrics (DCR, MCS, DDI), each
in a lexical (TF-IDF) and a semantic (Sentence-BERT) variant.

Script messages and the notes in the review sheet are in Vietnamese; file names, column
names and this README are in English.

## Repository contents

### Dataset

| File | Content |
|---|---|
| `labeled_adrs.json` | Final dataset: 88 ADRs from 39 repositories (`repo`, `path`, `url`, `context`, `decision`, `group`, `test_eligible`, label provenance) |
| `gold_test_set.json` | Held-out test set (35 ADRs) |
| `rag_pool.json` | Retrieval pool (53 ADRs) |
| `split_summary.csv`, `split_repo_overlap.csv` | ADRs per topic in each split; test ADRs that share a repository with pool ADRs |
| `labeled_adrs_before_review.json` | The 134 candidate records before review and re-extraction |
| `review_sheet_reviewed.csv` | Review of all 134 candidates: usable ADR or not, in scope or not, topic, and a justification per record |
| `finalize_report.json`, `finalize_detail.csv` | Outcome of the review and re-extraction for every record |
| `extraction_audit.csv` | Extraction audit of the 134 candidates against their source files |
| `collection_log.csv`, `dedupe_report.csv` | Collection and de-duplication logs |

Topic groups: G1 service decomposition, G2 communication, G3 data management, G4 API
gateway / discovery, G5 deployment / orchestration, G6 resilience, G7 observability /
security.

The `context` and `decision` fields are verbatim excerpts of ADRs published in public
GitHub repositories. Each record carries its source URL; the texts remain under the
licenses of their source repositories.

### Experiment outputs

| File | Content |
|---|---|
| `generation_results.json` | All 1,400 generations with prompt condition, retrieved examples, finish reason and token usage |
| `generation_failures.json` | Calls that failed after three attempts within a run (all were completed on a later run) |
| `evaluation_results.json` | One record per generation with BERTScore and DCR |
| `units.csv` | One row per (ADR, model, condition): means over the five runs and MCS |
| `table_*.csv`, `test_group_counts.csv` | Tables written by `3_evaluation_layer.py` |
| `paper_tables/` | Additional tables written by `3b_paper_tables.py` |

### Scripts

| Step | Script | Purpose |
|---|---|---|
| 1.1 | `1_1_collect_adrs_v2.py`, `..._batch2.py`, `..._batch3.py` | Scan a curated list of repositories (Git Tree API) |
| 1.1 | `1_1_collect_adrs_batch4_discovery.py`, `1_1_collect_adrs_batch5_targeted_v4.py` | Discover ADRs with GitHub Code Search, including topic-targeted queries |
| 1.3 | `1_3_label_groups_auto.py` | Keyword-based topic proposal (automatic or interactive) |
| 1.3 | `1_3b_dedupe.py` | Remove duplicates and near-duplicates |
| 1.3 | `1_3c_review.py export` | Export the review sheet |
| 1.3 | `1_3e_extraction_audit.py` | Audit the extracted fields against the source files |
| 1.3 | `1_3d_finalize_dataset.py` | Apply the review, correct topics, re-extract Context/Decision with the heading-level-aware rule |
| 1.4 | `1_4_split_dataset.py` | Stratified random split into test set and retrieval pool (fixed seed) |
| 2 | `2_generation_layer.py` | Generation with and without retrieval; checkpointed and resumable |
| 3 | `3_evaluation_layer.py` | BERTScore, DCR, MCS, DDI, bootstrap confidence intervals, paired tests |
| 3 | `3b_paper_tables.py`, `make_fig.py` | Pooled tables, pairwise model comparisons and Fig. 2 of the paper |
| – | `4_repo_count.py`, `check_groq_models.py` | Repository counts; list the models available to an API key |

## Reproducing the results

```
pip install -r requirements.txt
```

**Tables and figure only** (no API key, no model download; a few seconds):

```
python 3b_paper_tables.py
python make_fig.py
```

**Evaluation from the released generations** (downloads `roberta-large` and
`all-MiniLM-L6-v2` on first use; reuses BERTScore values cached in
`evaluation_results.json`):

```
python 3_evaluation_layer.py
python 3b_paper_tables.py
```

**Dataset review, audit and split from the 134 candidates:**

```
copy labeled_adrs_before_review.json labeled_adrs.json     (Windows; use cp on Linux/macOS)
python 1_3e_extraction_audit.py
python 1_3d_finalize_dataset.py
python 1_4_split_dataset.py
```

**Generation** (requires a Groq API key in the environment variable `GROQ_API_KEY`):

```
python 2_generation_layer.py
```

The script refuses to mix results produced with a different dataset split, temperature or
token limit, so move or rename `generation_results.json` before starting a new run. It
saves after every call and can be re-run until all 1,400 generations are present.

**Collection** (requires a GitHub token in `GITHUB_TOKEN`): run the `1_1_*` scripts in the
order listed above, then `1_3_label_groups_auto.py` and `1_3b_dedupe.py`.

## Experimental settings

| Setting | Value |
|---|---|
| Models (Groq) | `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, `allam-2-7b` |
| Temperature | 0.3 |
| Output limit | 1,024 completion tokens for every model |
| Reasoning | GPT-OSS: `reasoning_effort = low`; Qwen: `reasoning_effort = none` |
| Repetitions | 5 per (ADR, model, condition) |
| Retrieval | TF-IDF over Context fields, top 4, ADRs from the query's own repository excluded |
| BERTScore | `bert-score` default English model (`roberta-large`), no baseline rescaling |
| Sentence embeddings | `all-MiniLM-L6-v2`, maximum sequence length 512 |
| Confidence intervals | Percentile bootstrap over ADRs, 5,000 resamples, fixed seeds |

## Limitations of exact reproduction

- GitHub Code Search does not return the same results over time, and repositories change,
  so re-running the collection scripts will not reproduce the same 134 candidates. The
  released JSON files are the reference.
- Generation is stochastic (temperature 0.3) and hosted models change or are withdrawn.
  Two of the models used here replaced models that the provider retired shortly before
  the experiment. Re-running generation gives different texts; the released
  `generation_results.json` is the reference.
- Bootstrap intervals are reproducible only with the same NumPy version and seeds.

## License

Code is released under the terms in `LICENSE`. ADR texts in the dataset remain under the
licenses of their source repositories (see the `url` field of each record).
