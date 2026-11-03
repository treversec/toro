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


def test_ands_are_on_by_default():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert app.toggle(key="enable_upbeats").value is True
    assert app.slider(key="upbeat_factor").value == 2.0
    assert len(app.exception) == 0


def test_cycle_length_defaults_to_four_and_reaches_eight():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert app.slider(key="cycle_len").value == 4.0    # double the old 2.0s baseline
    assert app.slider(key="cycle_len").max == 8.0       # ceiling doubled too
    app.slider(key="cycle_len").set_value(6.0)
    app.run()
    assert len(app.exception) == 0


def test_spoken_counts_toggle_defaults_off():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert app.toggle(key="enable_counts").value is False
    assert len(app.exception) == 0


def test_accent_radios_default_to_first_pick_and_rerun_cleanly():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert app.radio(key="pyramid_accent").value == 2   # first-picked meter leads
    assert app.radio(key="fractal_accent").value == 2  # first-picked base leads
    app.radio(key="pyramid_accent").set_value(3)
    app.radio(key="fractal_accent").set_value(3)
    app.run()
    assert len(app.exception) == 0
