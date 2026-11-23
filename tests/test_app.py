"""Run the Streamlit app headlessly and check it renders without errors."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
needs_results = pytest.mark.skipif(
    not (ROOT / "data" / "processed" / "predictions.parquet").exists(),
    reason="run the pipeline first (python -m src.train)",
)


@needs_results
def test_app_renders():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title and "electricity" in at.title[0].value.lower()
    assert len(at.metric) >= 4
