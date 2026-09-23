"""English dashboard for an educational Steam recommender."""

import json
import pandas as pd
import streamlit as st
from steam_rec.data import demo_matrix, load_steam_csv
from steam_rec.model import predict, recommend, METHODS, SIMILARITIES
from steam_rec.evaluate import evaluate


@st.cache_data
def fit(frame, metric, threshold, weight):
    return predict(frame, metric, threshold, weight)


@st.cache_data
def benchmark(frame, metric, threshold, weight):
    return evaluate(frame, metric, threshold, weight)


def main():
    st.set_page_config(page_title="Steam Recommender Lab", page_icon="🎮", layout="wide")
    st.title("Steam Recommender Lab")
    st.caption(
        "Collaborative filtering · User-based, item-based and hybrid methods · German Kashirski"
    )
    if "matrix" not in st.session_state:
        st.session_state.matrix = demo_matrix()
        st.session_state.source = "Synthetic demonstration — invented players and games"
    with st.sidebar:
        st.header("Dataset")
        if st.button("Load synthetic demo"):
            st.session_state.matrix = demo_matrix()
            st.session_state.source = "Synthetic demonstration — invented players and games"
        upload = st.file_uploader("Steam-200k CSV (headerless)", type="csv")
        if upload is not None and st.button("Load uploaded data"):
            try:
                if upload.size > 20_000_000:
                    raise ValueError("Upload limit is 20 MB for this dense demonstration.")
                matrix = load_steam_csv(upload)
                st.session_state.matrix = matrix
                st.session_state.source = "Uploaded Steam data — playtime-derived engagement scores"
            except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
                st.error(str(exc))
        st.caption(
            "Up to 600 players and 100 games. Raw account IDs are replaced with local aliases."
        )
        st.header("Model")
        metric = st.selectbox("Similarity", SIMILARITIES)
        threshold = st.slider("Minimum neighbor similarity", 0.0, 1.0, 0.2, 0.05)
        weight = st.slider("User-based weight in hybrid", 0.0, 1.0, 0.5, 0.05)
    frame = st.session_state.matrix
    st.info(st.session_state.source)
    a, b, c = st.columns(3)
    a.metric("Players", len(frame))
    b.metric("Games", frame.shape[1])
    c.metric("Observed interactions", int(frame.notna().sum().sum()))
    rec_tab, eval_tab, data_tab = st.tabs(
        ["Recommendations", "Holdout evaluation", "Data and method"]
    )
    with rec_tab:
        mode = st.radio("Profile", ["Existing player", "New profile"], horizontal=True)
        if mode == "Existing player":
            user = st.selectbox("Player", frame.index.tolist())
            working = frame
        else:
            selected = st.multiselect(
                "Games to include in your profile", frame.columns.tolist(), max_selections=8
            )
            profile = pd.Series(float("nan"), index=frame.columns)
            for game in selected:
                profile[game] = st.slider(
                    f"Preference score: {game}", 0.5, 5.0, 3.0, 0.5, key=f"score_{game}"
                )
            user = "new-profile"
            working = pd.concat([frame, pd.DataFrame([profile], index=[user])])
            st.caption(
                "Manual preferences are illustrative and are not calibrated to playtime-derived scores. "
                "With no history, the model uses mean-score fallbacks."
            )
        method = st.selectbox("Recommendation method", METHODS, index=2)
        count = st.slider("Number of recommendations", 1, 20, 10)
        predictions = fit(working, metric, threshold, weight)
        ranked = recommend(working, predictions[method], user, count)
        st.subheader("Discover unobserved games")
        st.caption(
            "Estimated scores are on a 0.5–5 scale, not probabilities or verified star ratings."
        )
        st.dataframe(ranked, hide_index=True, width="stretch")
        if ranked.empty:
            st.info("No unobserved games remain for this profile.")
        st.download_button(
            "Download recommendations",
            ranked.to_csv(index=False),
            "recommendations.csv",
            "text/csv",
        )
        with st.expander("Observed profile"):
            st.dataframe(working.loc[user].dropna().rename("Score"), width="stretch")
    with eval_tab:
        st.write(
            "Seeded per-player holdout: hide 20% of observed interactions, at least one when possible, "
            "while retaining at least one training interaction per player."
        )
        st.caption(
            "All similarities, means and predictions are fitted on training values only. "
            "Evaluation uses the dataset, not the manually entered profile. This is a random "
            "warm-user experiment, not a chronological or unseen-user benchmark."
        )
        if st.button("Run holdout evaluation"):
            with st.spinner("Evaluating all four methods on the same split…"):
                try:
                    metrics, details = benchmark(frame, metric, threshold, weight)
                    st.session_state.evaluation = (
                        metrics,
                        details,
                        frame.copy(),
                        metric,
                        threshold,
                        weight,
                    )
                except ValueError as exc:
                    st.error(str(exc))
        saved = st.session_state.get("evaluation")
        if saved and saved[2].equals(frame) and saved[3:] == (metric, threshold, weight):
            metrics, details = saved[:2]
            st.dataframe(metrics, hide_index=True, width="stretch")
            st.bar_chart(metrics.set_index("Method")[["RMSE", "MAE"]])
            st.caption(
                "Lower RMSE/MAE is better; higher Recall@5/NDCG@5 is better. "
                "All held-out plays count as relevant for ranking; unobserved games are not proven dislikes."
            )
            st.json(details)
            st.download_button(
                "Download metrics", metrics.to_csv(index=False), "metrics.csv", "text/csv"
            )
            st.download_button(
                "Download evaluation settings",
                json.dumps(details, indent=2),
                "evaluation.json",
                "application/json",
            )
        else:
            st.info("Run evaluation for the current dataset and settings.")
    with data_tab:
        st.markdown("""**Pipeline:** play records → fixed log-playtime transform → user/game matrix →
        observed-interaction holdout → neighborhood models → unseen-game ranking.

        **User-based:** players with similar histories contribute deviations from their own means.
        **Item-based:** observed games contribute deviations from their game means.
        **Hybrid:** a weighted blend of the two. **Baseline:** training item means.

        Denominators include only neighbors that actually observed the relevant entry.
        No neighbors means a deterministic mean fallback. Pearson uses jointly observed entries;
        Jaccard compares which entries exist; cosine treats missing entries as zero only for similarity.

        The local CSV subset is chosen by interaction counts before splitting. Results are conditional
        on that selected population and catalog, not an unbiased benchmark of the full Steam catalog.
        Dense similarities are intended for small datasets. There is no Steam API integration.
        """)
        st.dataframe(frame.head(15), width="stretch")
    st.caption("Independent educational project. Not affiliated with Valve or Steam.")


if __name__ == "__main__":
    main()
