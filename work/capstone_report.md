# Capstone Report — CTR / Engagement Opportunity Scoring

- **Author:** Majhabul Hoque
- **Lane:** CTR / Engagement Opportunity Scoring
- **Repo:** https://github.com/Catacombs666/ML_int-
- **Paper:** https://catacombs666.github.io/ML_int-/
- **Report revision:** 2026-10-13

## 0. Abstract

This study asks which visible pages merit a click-capture review after observed position is considered. The permitted FlyRank starter snapshot supplies 30,000 pseudonymized pages, of which 16,726 meet the 500-impression and valid-position requirements. Separate client groups support fitting, validation-only method selection, and a final comparison of a pooled position baseline with Random Forest and histogram gradient boosting. Random Forest reduces impression-weighted CTR error by 15.5% on 3,913 pages from six test clients. The result is a reason-coded human review queue whose reference gaps do not estimate recoverable clicks, future CTR, or causal refresh benefit.

## 1. Problem framing

A content team must decide which already-visible pages to inspect first for click capture. One observation is a pseudonymized content item over a trailing 90-day snapshot. The model estimates an observed CTR reference and the output ranks positive shortfalls against that reference; a person then checks metadata, intent and search-result context. A false recommendation can waste editorial effort or damage a useful page. Position and safe metadata provide a more explicit comparison than labeling every low CTR page as defective.

## 2. Data safety

This core lane uses the bundled `data/raw/content_refresh_anonymized.csv`, explicitly permitted by the lane guide. It is **not** a claim to have queried the gated warehouse. The source has 30,000 rows, 44 fields, 32 clients and zero duplicate content IDs. A 500-impression floor excludes 13,274 pages; 16,726 remain across 28 clients. Positive measured position is required; position zero denotes unavailable measurement. There are no additional position exclusions after this floor.

The slice contains trailing 90-day aggregates but no absolute export/end dates, so warehouse calendar dates are not assigned to it. Word count is missing for 28.393% of eligible pages. Click counts must be nonnegative and not exceed impressions. CTR is recalculated from counts rather than rounded stored rates.

IDs are grouping/local-mapping context only. CTR, raw clicks, trend fields, engagement outcomes and target-derived features are excluded from the input vector. No client names, domains, private queries, credentials or new raw exports are committed. Public outputs consist of aggregates and limited derived, identifier-free diagnostics/recommendations. The row mapping is regenerated only in gitignored `work/.cache/review_queue.csv`.

Input SHA-256: `c43bdac4eccfa17fcd8a33974fe36f2c998c03a3ae3af8d80cf712abab5d6396`. Source Git blob: `2f2f41fdb07868dc5599449ce9d817f58f3b2ae9`; pinned source commit: `3b657dda40c1b668072a53371883fd4413d60f08`.

## 3. Baseline

The baseline estimates pooled CTR within observed position bands 1–3, 4–10, 11–20, 21–50 and >50. Each band is smoothed with 500 pseudo-impressions at training-wide CTR. The fitted reference is evaluated on exactly the same validation/test rows as both models; no test observations enter fitting. Its test weighted MAE is 0.2042 percentage points, and page-average MAE is 0.2013 percentage points. Test exposure-weighted observed CTR is 0.2850% — the relevant outcome level for this regression task, not a binary-class base rate.

## 4. Model / analysis

The target is `100 * clicks_90d / impressions_90d`: observed same-window CTR, not a future label or success-of-edit label.

Exact features: `avg_position, log_word_count, content_age_days, days_since_last_update, log_search_volume, competition, log_cpc, content_type, main_intent`. Imputation and categorical encoding are fitted inside training pipelines; numerical missingness indicators are retained. The models use square-root impression weights during training, while evaluation weights are actual impressions.

Random Forest: 200 trees, maximum depth 12, minimum leaf size 30. Histogram gradient boosting: 180 iterations, maximum 15 leaves, minimum leaf size 40, learning rate 0.06, L2 regularization 5. Both have seed 42. Random Forest is selected by validation weighted MAE, then refitted on the combined development clients before the test comparison. No selection is performed using test errors.

## 5. Evaluation

Client separation prevents the same client's pages from appearing in multiple folds: 8,483 rows / 17 initial training clients, 4,330 rows / 5 validation clients, and 3,913 rows / 6 test clients. Client/content overlap is zero. This is client generalization within a snapshot; it is not time-forward validation.

| Method | Validation weighted MAE (pp) | Test weighted MAE (pp) | Test page MAE (pp) | Test RMSE (pp) |
|---|---:|---:|---:|---:|
| position_baseline | 0.2102 | 0.2042 | 0.2013 | 0.2870 |
| random_forest | 0.2056 | 0.1726 | 0.1729 | 0.2742 |
| hist_gradient_boosting | 0.2079 | 0.1750 | 0.1711 | 0.2763 |

Weighted MAE = sum(impressions × absolute CTR residual) / sum(impressions). The selected method lowers the primary test error by 15.5%. However, the baseline has lower page-average validation MAE than either fitted model, and gradient boosting has lower page-average test MAE than Random Forest. The objective determines the preferred method.

Calibration, position/content-type subgroup errors, 500/1,000/3,000-impression sensitivity, and three derived error diagnostics are recorded in `outputs/ctr_metrics.json` and displayed in the paper/notebooks. Predicted weighted CTR for the selected model is 0.2641% versus observed 0.2850%, showing imperfect calibration. Large individual errors can reflect unavailable query mix, brand demand and SERP context.

A 2,000-replicate client bootstrap yields a selected-minus-baseline weighted MAE difference of -0.03160 pp, with exploratory 95% interval [-0.04499, -0.02152] pp. Only six clusters support it; it does not establish universal performance.

## 6. Interpretation

| Feature | Mean weighted MAE increase after permutation (pp) | Repeat SD (pp) |
|---|---:|---:|
| avg_position | 0.02815 | 0.00399 |
| content_age_days | 0.00378 | 0.00427 |
| log_word_count | 0.00291 | 0.00140 |
| log_search_volume | 0.00158 | 0.00075 |
| main_intent | 0.00106 | 0.00022 |
| competition | 0.00050 | 0.00021 |
| content_type | 0.00000 | 0.00000 |
| log_cpc | -0.00021 | 0.00008 |
| days_since_last_update | -0.00051 | 0.00060 |

Position has the largest measured permutation sensitivity. Age and word count have much smaller, less stable associations. Freshness and CPC show no positive improvement in this permutation test; therefore stale/short-content tags in the action rules are contextual cues, not validated causal editing prescriptions. Correlated inputs and systematic missingness can redistribute importance. These results do not identify Google ranking factors.

## 7. Recommendation

Reference gap = impressions × max(predicted CTR − observed CTR, 0) / 100. Rank descending on held-out pages. Generic review IDs map locally to pseudonymized items; confidence wording describes measurement volume, not probability that an edit will work.

| Rank | Review ID | Reference gap (click units) | Human review | Reason codes |
|---:|---|---:|---|---|
| 1 | Review 01 | 1403.3 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, word_count_missing, recent_update_context |
| 2 | Review 02 | 910.5 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, content_depth_context, recent_update_context |
| 3 | Review 03 | 465.6 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, content_depth_context, recent_update_context |
| 4 | Review 04 | 366.5 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, word_count_missing, recent_update_context |
| 5 | Review 05 | 339.7 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, content_depth_context, recent_update_context |
| 6 | Review 06 | 312.3 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, content_depth_context, recent_update_context |
| 7 | Review 07 | 276.1 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, content_depth_context, recent_update_context |
| 8 | Review 08 | 271.7 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, content_depth_context, recent_update_context |
| 9 | Review 09 | 231.8 | Visibility and content review | below_grouped_ctr_reference, outside_page_two, content_depth_context, recent_update_context |
| 10 | Review 10 | 230.7 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, content_depth_context, recent_update_context |
| 11 | Review 11 | 216.8 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, content_depth_context, recent_update_context |
| 12 | Review 12 | 211.6 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, content_depth_context, recent_update_context |
| 13 | Review 13 | 191.6 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, word_count_missing, recent_update_context |
| 14 | Review 14 | 187.7 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, content_depth_context, recent_update_context |
| 15 | Review 15 | 186.9 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, content_depth_context, recent_update_context |
| 16 | Review 16 | 184.2 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, content_depth_context, recent_update_context |
| 17 | Review 17 | 167.0 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, content_depth_context, recent_update_context |
| 18 | Review 18 | 165.0 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, word_count_missing, recent_update_context |
| 19 | Review 19 | 161.3 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, word_count_missing, recent_update_context |
| 20 | Review 20 | 157.1 | Metadata and intent review | below_grouped_ctr_reference, visible_position_band, word_count_missing, recent_update_context |

For visible positions ≤20, inspect metadata and intent/SERP alignment. Outside page two, inspect visibility/content coverage and demand context. Verify missing measurement fields before interpreting depth. Do not prune, rewrite or promise uplift from a score alone. Query/SERP/brand context, benign low-click intent, measurement gaps and reference bias can make a recommendation wrong. Any approved intervention needs an observation window and comparison design before a causal claim.

## 8. Reproducibility

```bash
git clone https://github.com/Catacombs666/ML_int-.git
cd ML_int-
python -m pip install -r work/requirements.txt
python work/scripts/ctr_capstone.py
python work/scripts/build_deliverables.py
python work/scripts/run_notebooks.py
python work/scripts/verify_submission.py
```

All custom work lives under `work/`; root `scripts/` remains the supplied reference pipeline. The Python-only runner executes notebook cells in order and records genuine outputs; Colab/Jupyter can run the same Python cells. Rebuilding notebook templates resets outputs, so run the notebook runner afterward. Guided starter notebooks 01/02 and the legacy root Week-2 copy are also executed by the runner.

Seed: 42. Measured environment: Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0. Dependency pins are in `work/requirements.txt`. Frame construction, split logic and validation selection are in `work/scripts/ctr_capstone.py`; the measured receipt is `work/outputs/ctr_metrics.json`. The exact live paper URL is the one-line `submission/paper_url.txt`.

The separate full-warehouse tutorial remains access-dependent: this environment has no `HF_TOKEN`. Its queries and models have not been reported as executed. Gated data access is not required for the permitted starter-data CTR capstone.

## 9. Acknowledgments & data credit

**Built on the FlyRank ML Internship dataset** — https://flyrank.ai. The original reference pipeline and data dictionary are credited as internship resources. Implementation was AI-assisted and checked against executed code/outputs; the author should review the findings and submit the repository URL through the internship portal.
