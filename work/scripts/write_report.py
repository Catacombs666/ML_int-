"""Generate the full rubric-aligned report from measured CTR receipts."""
from pathlib import Path
import json

def write_report(r,root):
    root=Path(root);b=r['test']['position_baseline'];s=r['test'][r['selected_method']];gain=100*(1-s['weighted_mae_pp']/b['weighted_mae_pp'])
    table='\n'.join(f"| {name} | {r['validation'][name]['weighted_mae_pp']:.4f} | {m['weighted_mae_pp']:.4f} | {m['mae_pp']:.4f} | {m['rmse_pp']:.4f} |" for name,m in r['test'].items())
    importance='\n'.join(f"| {x['feature']} | {x['weighted_mae_increase_pp']:.5f} | {x['sd_pp']:.5f} |" for x in sorted(r['permutation_importance'],key=lambda x:x['weighted_mae_increase_pp'],reverse=True))
    queue='\n'.join(f"| {x['rank']} | {x['review_id']} | {x['reference_gap_clicks']:.1f} | {x['action']} | {', '.join(x['reason_codes'])} |" for x in r['recommendations'])
    text=f'''# Capstone Report — CTR / Engagement Opportunity Scoring

- **Author:** Majhabul Hoque
- **Lane:** CTR / Engagement Opportunity Scoring
- **Repo:** https://github.com/Catacombs666/ML_int-
- **Paper:** https://catacombs666.github.io/ML_int-/
- **Report revision:** 2026-10-13

## 0. Abstract

This study asks which visible pages merit a click-capture review after observed position is considered. The permitted FlyRank starter snapshot supplies 30,000 pseudonymized pages, of which 16,726 meet the 500-impression and valid-position requirements. Separate client groups support fitting, validation-only method selection, and a final comparison of a pooled position baseline with Random Forest and histogram gradient boosting. Random Forest reduces impression-weighted CTR error by {gain:.1f}% on 3,913 pages from six test clients. The result is a reason-coded human review queue whose reference gaps do not estimate recoverable clicks, future CTR, or causal refresh benefit.

## 1. Problem framing

A content team must decide which already-visible pages to inspect first for click capture. One observation is a pseudonymized content item over a trailing 90-day snapshot. The model estimates an observed CTR reference and the output ranks positive shortfalls against that reference; a person then checks metadata, intent and search-result context. A false recommendation can waste editorial effort or damage a useful page. Position and safe metadata provide a more explicit comparison than labeling every low CTR page as defective.

## 2. Data safety

This core lane uses the bundled `data/raw/content_refresh_anonymized.csv`, explicitly permitted by the lane guide. It is **not** a claim to have queried the gated warehouse. The source has 30,000 rows, 44 fields, 32 clients and zero duplicate content IDs. A 500-impression floor excludes 13,274 pages; 16,726 remain across 28 clients. Positive measured position is required; position zero denotes unavailable measurement. There are no additional position exclusions after this floor.

The slice contains trailing 90-day aggregates but no absolute export/end dates, so warehouse calendar dates are not assigned to it. Word count is missing for 28.393% of eligible pages. Click counts must be nonnegative and not exceed impressions. CTR is recalculated from counts rather than rounded stored rates.

IDs are grouping/local-mapping context only. CTR, raw clicks, trend fields, engagement outcomes and target-derived features are excluded from the input vector. No client names, domains, private queries, credentials or new raw exports are committed. Public outputs consist of aggregates and limited derived, identifier-free diagnostics/recommendations. The row mapping is regenerated only in gitignored `work/.cache/review_queue.csv`.

Input SHA-256: `{r['data']['sha256']}`. Source Git blob: `{r['data']['source_git_blob']}`; pinned source commit: `{r['data']['source_commit']}`.

## 3. Baseline

The baseline estimates pooled CTR within observed position bands 1–3, 4–10, 11–20, 21–50 and >50. Each band is smoothed with 500 pseudo-impressions at training-wide CTR. The fitted reference is evaluated on exactly the same validation/test rows as both models; no test observations enter fitting. Its test weighted MAE is {b['weighted_mae_pp']:.4f} percentage points, and page-average MAE is {b['mae_pp']:.4f} percentage points. Test exposure-weighted observed CTR is {b['observed_ctr_pp']:.4f}% — the relevant outcome level for this regression task, not a binary-class base rate.

## 4. Model / analysis

The target is `100 * clicks_90d / impressions_90d`: observed same-window CTR, not a future label or success-of-edit label.

Exact features: `{', '.join(r['feature_columns'])}`. Imputation and categorical encoding are fitted inside training pipelines; numerical missingness indicators are retained. The models use square-root impression weights during training, while evaluation weights are actual impressions.

Random Forest: 200 trees, maximum depth 12, minimum leaf size 30. Histogram gradient boosting: 180 iterations, maximum 15 leaves, minimum leaf size 40, learning rate 0.06, L2 regularization 5. Both have seed 42. Random Forest is selected by validation weighted MAE, then refitted on the combined development clients before the test comparison. No selection is performed using test errors.

## 5. Evaluation

Client separation prevents the same client's pages from appearing in multiple folds: 8,483 rows / 17 initial training clients, 4,330 rows / 5 validation clients, and 3,913 rows / 6 test clients. Client/content overlap is zero. This is client generalization within a snapshot; it is not time-forward validation.

| Method | Validation weighted MAE (pp) | Test weighted MAE (pp) | Test page MAE (pp) | Test RMSE (pp) |
|---|---:|---:|---:|---:|
{table}

Weighted MAE = sum(impressions × absolute CTR residual) / sum(impressions). The selected method lowers the primary test error by {gain:.1f}%. However, the baseline has lower page-average validation MAE than either fitted model, and gradient boosting has lower page-average test MAE than Random Forest. The objective determines the preferred method.

Calibration, position/content-type subgroup errors, 500/1,000/3,000-impression sensitivity, and three derived error diagnostics are recorded in `outputs/ctr_metrics.json` and displayed in the paper/notebooks. Predicted weighted CTR for the selected model is {s['predicted_ctr_pp']:.4f}% versus observed {s['observed_ctr_pp']:.4f}%, showing imperfect calibration. Large individual errors can reflect unavailable query mix, brand demand and SERP context.

A 2,000-replicate client bootstrap yields a selected-minus-baseline weighted MAE difference of {r['bootstrap']['difference_selected_minus_baseline_pp']:.5f} pp, with exploratory 95% interval [{r['bootstrap']['client_bootstrap_95pct_interval_pp'][0]:.5f}, {r['bootstrap']['client_bootstrap_95pct_interval_pp'][1]:.5f}] pp. Only six clusters support it; it does not establish universal performance.

## 6. Interpretation

| Feature | Mean weighted MAE increase after permutation (pp) | Repeat SD (pp) |
|---|---:|---:|
{importance}

Position has the largest measured permutation sensitivity. Age and word count have much smaller, less stable associations. Freshness and CPC show no positive improvement in this permutation test; therefore stale/short-content tags in the action rules are contextual cues, not validated causal editing prescriptions. Correlated inputs and systematic missingness can redistribute importance. These results do not identify Google ranking factors.

## 7. Recommendation

Reference gap = impressions × max(predicted CTR − observed CTR, 0) / 100. Rank descending on held-out pages. Generic review IDs map locally to pseudonymized items; confidence wording describes measurement volume, not probability that an edit will work.

| Rank | Review ID | Reference gap (click units) | Human review | Reason codes |
|---:|---|---:|---|---|
{queue}

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

Seed: {r['seed']}. Measured environment: Python {r['versions']['python']}, NumPy {r['versions']['numpy']}, pandas {r['versions']['pandas']}, scikit-learn {r['versions']['sklearn']}. Dependency pins are in `work/requirements.txt`. Frame construction, split logic and validation selection are in `work/scripts/ctr_capstone.py`; the measured receipt is `work/outputs/ctr_metrics.json`. The exact live paper URL is the one-line `submission/paper_url.txt`.

The separate full-warehouse tutorial remains access-dependent: this environment has no `HF_TOKEN`. Its queries and models have not been reported as executed. Gated data access is not required for the permitted starter-data CTR capstone.

## 9. Acknowledgments & data credit

**Built on the FlyRank ML Internship dataset** — https://flyrank.ai. The original reference pipeline and data dictionary are credited as internship resources. Implementation was AI-assisted and checked against executed code/outputs; the author should review the findings and submit the repository URL through the internship portal.
'''
    (root/'work/capstone_report.md').write_text(text)
    return text
if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]
    write_report(json.loads((root/'work/outputs/ctr_metrics.json').read_text()),root)
