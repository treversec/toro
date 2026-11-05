"""The full app script must run cleanly in Streamlit's runtime.

Widget identifiers are auto-generated from type + label, so two buttons with the
same label (for example the pyramid's and the fractal's "Play reversed") collide
with a `StreamlitDuplicateElementId` unless each carries a unique `key`.
"""
from pathlib import Path

import toro
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


def test_cycle_length_defaults_to_two_and_reaches_eight():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert app.slider(key="cycle_len").value == 2.0    # a snappy two-second baseline
    assert app.slider(key="cycle_len").max == 8.0      # ceiling stays at eight seconds
    app.slider(key="cycle_len").set_value(6.0)
    app.run()
    assert len(app.exception) == 0


def test_spoken_counts_toggle_defaults_off():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert app.toggle(key="enable_counts").value is False
    assert len(app.exception) == 0


def test_alternate_option_sits_on_each_accent_radio():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    # Both accent radios offer the rotation option after the single bases
    # (listed by its formatted label; selected by its sentinel value).
    alt_label = toro._accent_label(toro.ALTERNATE_ACCENT)
    assert alt_label in app.radio(key="pyramid_accent").options
    assert alt_label in app.radio(key="fractal_accent").options
    # Picking it on the pyramid radio surfaces that section's turn slider (2 by
    # default) — and the fractal one stays independent with its own key.
    app.radio(key="pyramid_accent").set_value(toro.ALTERNATE_ACCENT)
    app.run()
    assert app.slider(key="turn_len_pyramid").value == 2
    app.radio(key="fractal_accent").set_value(toro.ALTERNATE_ACCENT)
    app.run()
    assert app.slider(key="turn_len_fractal").value == 2
    assert len(app.exception) == 0
    # The old sidebar toggle is gone — there is no global rotate switch anymore.
    assert not any(w.key == "rotate_accent" for w in app.toggle)


def test_accent_radios_default_to_first_pick_and_rerun_cleanly():
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert app.radio(key="pyramid_accent").value == 2   # first-picked meter leads
    assert app.radio(key="fractal_accent").value == 2  # first-picked base leads
    app.radio(key="pyramid_accent").set_value(3)
    app.radio(key="fractal_accent").set_value(3)
    app.run()
    assert len(app.exception) == 0
