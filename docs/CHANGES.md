# Changes from the coursework archive

Source: `prepare_steam_data.py` and `prac1/comparison.ipynb` in the local seventh-semester
recommendation-systems archive. Original files were read, not modified. The archived
matrix was 2,436 players × 639 games.

## Reproduced validation defect

The notebook sampled 20% of matrix **rows** as a validation mask. This creates a 487-row
mask with original labels, but then `np.where(mask.values)` produces positional indices
relative to those 487 rows. Applying those positions to the full matrix points at different
users. Reproduction on the archived matrix found **9,719 / 9,719 held-out coordinates**
associated with different user identities. Old reported metrics cannot be relied on.

The entire original notebook was not rerun end to end; the defective split/index logic
was reproduced directly against its archived input matrix. The corrected workflow was
executed end to end on both the demonstration and raw CSV.

## Corrections

- Hold out individual observed interactions using a full-shape, label-aligned mask.
- Keep at least one observed training value for every evaluated player.
- Fit all means and similarities from training values, not held-out scores.
- Replace `np.divide(..., where=...)` without an output buffer with explicitly zero-filled
  safe division and documented fallback means.
- Normalize each prediction using only contributing observed neighbors.
- Use one consistent 0.5–5 score scale; the original preparation allowed 0.5 while
  predictions clipped to a minimum of 1.
- Replace sample-maximum playtime normalization with a fixed documented transform.
- Compute Pearson from jointly observed values instead of treating missing values as zero.
- Include an item-mean baseline and ranking metrics, not only regression metrics.
- Preserve deterministic ties, splits and recommendations.
- Validate input instead of silently skipping malformed rows.
- Bound the user/item subset before allocating dense matrices.

## Presentation and scope

The notebook now demonstrates shared library functions rather than duplicating algorithms.
The new English Streamlit interface supports player selection, a new illustrative profile,
settings, explicit evaluation and downloads. CLI outputs replace working-directory-dependent
chains of intermediate CSV files. English game names are retained rather than translated.

The original precomputed train/validation user files are not reused. Dependencies are kept
minimal: NumPy/pandas for algorithms and Streamlit for presentation; no trained neural model
is claimed. A synthetic example makes the repository independently runnable without bundling
the unverified-license dataset.

## Remaining limitations

Dense all-pairs similarities do not scale to the full Steam population. The selected subset
is popularity/activity biased. Missing interactions are not negative feedback. Random
holdout cannot measure future behavior, and playtime is not explicit satisfaction. Pearson
with two shared games can overfit; no confidence weighting or shrinkage is included.
Manual preferences and playtime scores have different semantics. Production deployment,
Steam authentication, content features and a dedicated final test split are out of scope.
