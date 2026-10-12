# Completed CTR opportunity capstone

Question: which visible pages merit a human click-capture review after position and safe metadata are considered?

Lane: **CTR / Engagement Opportunity Scoring**. The internship lane guide explicitly permits the bundled anonymized starter snapshot. This study is cross-sectional, not a future-growth forecast.

## Run from a fresh clone

```bash
python -m pip install -r work/requirements.txt
python work/scripts/ctr_capstone.py
python work/scripts/build_deliverables.py
python work/scripts/run_notebooks.py
python work/scripts/verify_submission.py
```

The Python-only notebook runner executes each cell in order and records actual text outputs. Colab/Jupyter Run All executes the same cells; a fresh Colab session automatically clones this repository if needed and installs the pinned work dependencies. Run the notebook runner after rebuilding notebook templates, because rebuilding intentionally resets their outputs.

## Assignment index

All ten notebooks were executed top to bottom; completed outputs are committed.

| Notebook | Assignment | Status |
|---|---|---|
| [Research question](notebooks/w01_research_question.ipynb) | ML-02 | Executed |
| [Task framing](notebooks/w02_ml_task_framing.ipynb) | ML-03 | Executed |
| [Data contract](notebooks/w03_data_contract.ipynb) | ML-04 | Executed |
| [Feature leakage checks](notebooks/w03_feature_leakage_check.ipynb) | ML-05 optional | Executed |
| [Signal audit](notebooks/w04_signal_audit.ipynb) | ML-06 optional | Executed |
| [Baseline and score](notebooks/w04_baseline_score.ipynb) | ML-07 | Executed |
| [Model](notebooks/w05_model.ipynb) | ML-08 | Executed |
| [Validation audit](notebooks/w06_validation_audit.ipynb) | ML-09 | Executed |
| [Action playbook](notebooks/w07_action_playbook.ipynb) | ML-10 | Executed |
| [Capstone](notebooks/capstone.ipynb) | ML-11 + ML-12 closing cells | Executed |

## Artifacts and boundaries

- `outputs/ctr_metrics.json`: aggregate metrics, provenance, diagnostics and limited derived recommendations; no raw exports or client identities.
- `figures/`: aggregate charts.
- `paper/`: public research paper source and its relative assets.
- `scripts/`: all custom implementation; the repository's root `scripts/` reference pipeline is unchanged.
- `.cache/review_queue.csv`: regenerated local mapping from generic review IDs to pseudonymous content items. It is ignored by git and never published.
- `../submission/paper_url.txt`: exact live paper URL, one line.

Observed reference shortfalls support human review. They do not measure recoverable clicks, future CTR, or causal refresh impact. The original solar-themed framing and generic dummy drafts are retained in repository history; the current assignment sequence is aligned to real FlyRank data.

Built on the FlyRank ML Internship dataset — https://flyrank.ai.

## Author and remaining access requirement

Author: **Majhabul Hoque**. The guided Week 1 and Week 2 exercises and all ten assignment/capstone notebooks are completed with outputs. The separate full-warehouse tutorial (`notebooks/03_working_with_the_full_release.ipynb`) remains unexecuted pending authorized Hugging Face access; no warehouse execution is claimed.
