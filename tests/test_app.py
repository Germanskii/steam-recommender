from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_app_evaluation_and_profile():
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30
    ).run()
    assert not app.exception and app.metric[0].value == "48"
    next(b for b in app.button if b.label == "Run holdout evaluation").click().run()
    assert not app.exception
    assert len(app.session_state.evaluation[0]) == 4
    next(r for r in app.radio if r.label == "Profile").set_value("New profile").run()
    assert not app.exception
    app.multiselect[0].set_value(["Orbit Quest"]).run()
    assert not app.exception
    assert "Orbit Quest" not in app.dataframe[0].value["Game"].tolist()
    next(s for s in app.selectbox if s.label == "Similarity").set_value("pearson").run()
    assert not app.exception
    assert any("Run evaluation for the current" in x.value for x in app.info)
