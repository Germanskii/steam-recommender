import io
import numpy as np
import pandas as pd
import pytest
from steam_rec.data import demo_matrix, load_steam_csv
from steam_rec.model import predict, recommend, similarity, validate_matrix
from steam_rec.evaluate import split_observed, evaluate


def test_holdout_alignment_and_no_leakage():
    frame = demo_matrix().sample(frac=1, random_state=7)
    train, mask = split_observed(frame)
    assert mask.index.equals(frame.index) and mask.columns.equals(frame.columns)
    assert train.to_numpy()[mask.to_numpy()].tolist() != []
    assert np.isnan(train.to_numpy()[mask.to_numpy()]).all()
    assert train.notna().sum(axis=1).ge(1).all()
    assert (mask & frame.isna()).to_numpy().sum() == 0
    altered = frame.copy()
    altered[mask] = 5.0
    train_again, mask_again = split_observed(altered)
    pd.testing.assert_frame_equal(train, train_again)
    pd.testing.assert_frame_equal(mask, mask_again)
    assert int(mask.to_numpy().sum()) == 48


@pytest.mark.parametrize("metric", ["cosine", "jaccard", "pearson", "mixed"])
def test_predictions_and_unseen_recommendations(metric):
    frame = demo_matrix()
    for scores in predict(frame, metric).values():
        assert np.isfinite(scores.to_numpy()).all()
        assert scores.to_numpy().min() >= 0.5 and scores.to_numpy().max() <= 5
        recs = recommend(frame, scores, frame.index[0])
        assert not set(recs.Game).intersection(frame.loc[frame.index[0]].dropna().index)
        assert recs["Estimated score"].is_monotonic_decreasing


def test_missing_neighbor_denominator():
    frame = pd.DataFrame([[2.0, np.nan, np.nan], [4.0, 5.0, np.nan], [1.0, np.nan, 4.0]])
    scores = predict(frame, threshold=0)["User-based"]
    # Only the second user observed item 1: centered deviation is +0.5.
    assert scores.iloc[0, 1] == pytest.approx(2.5)


def test_no_neighbors_has_defined_fallback():
    frame = pd.DataFrame([[0.5, np.nan], [np.nan, 5.0]])
    scores = predict(frame, threshold=1)
    np.testing.assert_allclose(scores["User-based"], [[0.5, 0.5], [5, 5]])
    np.testing.assert_allclose(scores["Item-based"], [[0.5, 5], [0.5, 5]])


def test_pearson_uses_only_overlap():
    values = np.array([[1.0, 2.0, 3.0, np.nan], [2.0, 4.0, 6.0, 1.0], [3.0, 2.0, 1.0, np.nan]])
    result = similarity(values, "pearson")
    assert result[0, 1] == pytest.approx(1)
    assert result[0, 2] == 0
    assert np.diag(result).sum() == 0


def test_evaluation_is_reproducible():
    a, details = evaluate(demo_matrix())
    b, other = evaluate(demo_matrix())
    pd.testing.assert_frame_equal(a, b)
    assert details == other and details["held_interactions"] == 48
    assert a[["Recall@5", "NDCG@5"]].ge(0).all().all()
    assert a[["Recall@5", "NDCG@5"]].le(1).all().all()


def test_csv_import_filters_purchase_and_deduplicates():
    csv = "1,A,play,10,0\n1,A,play,10,0\n1,B,play,20,0\n2,A,play,15,0\n2,B,play,25,0\n2,C,purchase,1,0\n"
    frame = load_steam_csv(io.StringIO(csv))
    assert frame.shape == (2, 2)
    assert list(frame.index) == ["player-0001", "player-0002"]
    assert frame.iloc[0, 0] == pytest.approx(0.5 + 4.5 * np.log1p(10) / np.log1p(1000))


@pytest.mark.parametrize("csv", ["a,b\n1,2", "1,A,purchase,1,0", "1,A,play,-2,0", "1,A,play,inf,0"])
def test_invalid_csv(csv):
    with pytest.raises(ValueError):
        load_steam_csv(io.StringIO(csv))


def test_cold_user_and_item_are_finite():
    frame = pd.DataFrame([[1.0, 2.0, np.nan], [np.nan, np.nan, np.nan]])
    for pred in predict(frame).values():
        assert np.isfinite(pred.to_numpy()).all()


def test_invalid_scale_rejected():
    with pytest.raises(ValueError):
        validate_matrix(pd.DataFrame([[0.0, 6.0]]))
