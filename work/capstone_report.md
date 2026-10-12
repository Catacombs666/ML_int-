# Where visibility under-captures clicks

This study asks which visible pages merit a click-capture review after accounting for observed search position. Using the FlyRank anonymized starter snapshot, we built a position-band reference and two regression models on 16,726 eligible pages, with separate client groups for training, validation, and testing. Random Forest was selected on validation error and reduced impression-weighted CTR MAE from 0.2042 to 0.1726 percentage points on 3,913 pages from six held-out clients, a 15.5% reduction. The resulting ranked queue identifies reference click shortfalls and assigns reason-coded human review actions, while calibration and subgroup checks expose where the reference can be wrong. These are observed decision-support recommendations from one cross-sectional snapshot, with no claim of future performance, recoverable clicks, or causal refresh benefit.

## Scope
Permitted starter snapshot, observed CTR association, disjoint clients. Absolute snapshot dates are unavailable.

## Results receipt
See `outputs/ctr_metrics.json` and the deployed paper recorded in `../submission/paper_url.txt`. All public paper source and assets are in `paper/`.

## Reproduce
```bash
python -m pip install -r work/requirements.txt
python work/scripts/ctr_capstone.py
python work/scripts/build_deliverables.py
python work/scripts/run_notebooks.py
```

Built on the FlyRank ML Internship dataset: https://flyrank.ai
