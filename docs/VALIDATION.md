# Validation record

Date: **2026-09-23**. Local environment: Windows, Python 3.10.11, fresh project virtual
environment with pinned direct dependencies from the requirements files.

## Original baseline

The archive's existing matrix was 2,436 × 639. Reproducing its validation row-sampling
and positional indexing showed 9,719 of 9,719 selected positions referenced different
user identities. The original notebook's reported metrics were not reused. The entire
original notebook was not executed; this specific defect was reproduced on its input.

## Local checks

- `python -m pytest -q`: **17 passed**.
- Ruff lint and formatting checks passed; `pip check` found no broken dependencies.
- Synthetic CLI ran end to end, exporting recommendations, metrics and settings.
- Local Steam-200k CSV import and CLI evaluation completed end to end.
- All code cells in the English notebook executed sequentially with shared namespace;
  all four similarity choices completed. Notebook outputs remain cleared for publication.

Tests include label-aligned holdout, no held-out score influence on the training matrix,
retained player history, all similarity choices, a hand-calculated neighbor-denominator
example, deterministic no-neighbor fallback, consistent score bounds, overlap-only Pearson,
cold rows/columns, input rejection, reproducibility, unseen-item exclusion and UI interaction.

## Demonstration

48 invented players, 24 invented games, 384 observations, 48 held out. Cosine defaults:

| Method | RMSE | MAE | Recall@5 | NDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| User-based | 0.922509 | 0.711756 | 0.750000 | 0.499050 |
| Item-based | 0.779536 | 0.558060 | 0.666667 | 0.417951 |
| Hybrid | 0.719042 | 0.566936 | 0.708333 | 0.504827 |
| Item-mean baseline | 1.369058 | 1.328679 | 0.250000 | 0.143686 |

These deliberately structured synthetic preferences are not real Steam behavior.

## Local real-data integration

The archive's raw CSV was loaded locally, capped to 600 players and 100 games:
12,575 observations, 2,267 held out, 600 evaluated players, zero entirely cold training
items. Parameters: seed 42, holdout fraction 0.2, cosine, neighbor threshold 0.2,
hybrid user weight 0.5, ranking k=5. Aggregate metrics are recorded in the README.
Raw data and the original account IDs are not redistributed.

The selected catalog/player population depends on full-data observation counts. This
is explicitly a conditional warm-user evaluation, not an unbiased full-catalog estimate
or an untouched final test. No recommendation-quality guarantee follows from these checks.

## Browser and UI

Headless Microsoft Edge with Playwright against the actual local Streamlit server:

- Synthetic dashboard and recommendation table rendered; screenshot inspected.
- Recommendation CSV downloaded successfully.
- Holdout evaluation completed and metrics CSV downloaded.
- Real CSV upload and evaluation completed through the interface.
- Malformed CSV produced an English error without a Streamlit exception in a separate check.
- New-profile input, exclusion of selected games, similarity switching and stale-result
  hiding were exercised through Streamlit AppTest.

The first combined browser script timed out while waiting for the malformed-upload message
after replacing an existing upload. The isolated malformed-upload check passed. No claim
is made that all browser timings or every interaction sequence were exhaustively tested.

Browser checks are preparation-time automation, not part of the committed pytest suite.
GitHub Actions runs the offline checks and CLI on Linux with Python 3.10 and 3.12; see
Actions for the actual remote result. macOS and other browsers were not tested locally.
