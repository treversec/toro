"""The full app script must run cleanly in Streamlit's runtime.

Widget identifiers are auto-generated from type + label, so two buttons with the
same label (for example the pyramid's and the fractal's "Play reversed") collide
with a `StreamlitDuplicateElementId` unless each carries a unique `key`.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "toro.py")


def test_app_runs_without_exceptions():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert len(app.exception) == 0
