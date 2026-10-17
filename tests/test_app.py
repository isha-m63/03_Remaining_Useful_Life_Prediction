"""The Streamlit app must actually render (a syntax/import error would otherwise show up as an empty page)."""
from streamlit.testing.v1 import AppTest


def test_app_renders_with_metrics():
    at = AppTest.from_file("streamlit_app.py", default_timeout=120).run()
    assert not at.exception, [e.message for e in at.exception]
    assert [t.value for t in at.title], "app rendered no title"
    labels = [m.label for m in at.metric]
    assert "Predicted RUL (cycles)" in labels and "RMSE on 100 test engines" in labels
