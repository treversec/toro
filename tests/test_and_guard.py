"""The accent meter's "and" keeps a loudness floor that outlives normalization.

In a stack of bases, the mix peak (beats + crowns + waves) grows and the final
0.9 normalization crushes every quiet voice — the lead's "and" sagged from ~0.33
(solo) to ~0.24 (trio), which is why it got hard to make out. Each non-accent
meter keeps its own identity "and" (its latch of the pitch ladder), but the
accent meter's upbeats are re-stamped at AND_FLOOR × the mix peak before the
normalize, so the lead's "and" lands at AND_FLOOR in the finished clip whatever
the stack size — still lighter than its click, still higher pitched.

Every test declares the session state it needs and renders explicit short,
deterministic buffers; no network, no cache.
"""
import numpy as np
import streamlit as st

import toro

SR = 44100


def _and_peak(clip, t, half=0.02):
    """Max |sample| in a ±half-second window around t — the composite the ear
    hears at that moment (the "and" plus whatever background rides with it)."""
    lo, hi = max(0, int((t - half) * SR)), min(len(clip), int((t + half) * SR))
    return float(np.max(np.abs(clip[lo:hi])))


def _clip(meters, order_override=None):
    """The normalized one-cycle pyramid mix for `meters` (selection order =
    meters list order, so meters[0] is the accent / foundation)."""
    st.session_state["enable_upbeats"] = True
    st.session_state["waves"] = True
    st.session_state["enable_counts"] = False
    return toro.render_combined(list(meters), 2.0, 55, sr=SR)


def test_accent_and_keeps_a_loudness_floor_in_stacks():
    # The lead's "and" (t = 0.5 s is accent meter 2's upbeat) must not sink as
    # more bases play: solo, duo, and trio all keep it at ~AND_FLOOR of the mix.
    for order in ([2], [2, 3], [2, 3, 5]):
        clip = _clip(order)
        assert _and_peak(clip, 0.5) >= toro.AND_FLOOR - 0.05


def test_accent_and_does_not_sag_as_meters_stack():
    # The old behaviour dropped the lead's "and" by ~25% from solo to trio.
    # The floor keeps the duo/trio reading essentially flat versus solo.
    solo = _and_peak(_clip([2]), 0.5)
    trio = _and_peak(_clip([2, 3, 5]), 0.5)
    assert trio >= 0.85 * solo


def test_accent_and_landing_is_still_lighter_than_its_click():
    # The floor raises the "and" without letting it overtake the beat: the
    # accent click rides the 0.9 normalized ceiling, the "and" stays under it.
    clip = _clip([2, 3, 5])
    and_win = _and_peak(clip, 0.5)
    click_win = _and_peak(clip, 0.0, half=0.012)
    assert click_win > 0.8
    assert and_win < 0.7
    assert and_win < click_win


def test_fractal_accent_and_keeps_the_floor():
    st.session_state["enable_upbeats"] = True
    st.session_state["waves"] = True
    st.session_state["enable_counts"] = False
    roots = toro.build_fractal([2, 3], 1, {})  # accent = roots[0] = meter 2
    forest = toro.render_forest(roots, 1.0, 55, sr=SR)
    assert _and_peak(forest, 0.25) >= toro.AND_FLOOR - 0.05   # meter 2's "and"
    assert _and_peak(forest, 0.75) >= toro.AND_FLOOR - 0.05
    assert float(np.max(np.abs(forest))) == 0.9  # the forest still peaks like a cycle


def test_fractal_and_floor_holds_through_nested_layers():
    # Deep nesting stacks click-transients so high that the un-normalized peak
    # dwarfs a single "and"; the floor must still lift the accent's upbeats to
    # AND_FLOOR (with all waves off, nothing else keeps the moment loud).
    st.session_state["enable_upbeats"] = True
    st.session_state["waves"] = False
    st.session_state["enable_counts"] = False
    roots = toro.build_fractal([2, 3], 2, {2: {0}, 3: {0}})
    forest = toro.render_forest(roots, 1.0, 55, sr=SR)
    assert _and_peak(forest, 0.25) >= toro.AND_FLOOR - 0.05
    assert _and_peak(forest, 0.75) >= toro.AND_FLOOR - 0.05
    assert float(np.max(np.abs(forest))) == 0.9


def test_rotation_carries_the_floor_with_the_lead():
    st.session_state["enable_upbeats"] = True
    st.session_state["waves"] = True
    st.session_state["enable_counts"] = False
    clip = toro.render_rotated([2, 3], 1.0, 55, sr=SR, turn_len=1, accent=2)
    # Turn 1 (0–1 s, accent 2): its "and"s at 0.25/0.75 sit at the floor.
    assert _and_peak(clip, 0.25) >= toro.AND_FLOOR - 0.05
    assert _and_peak(clip, 0.75) >= toro.AND_FLOOR - 0.05
    # Turn 2 (1–2 s, accent 3): its "and"s at 1.1667/1.5/1.8333 carry the floor.
    assert _and_peak(clip, 1.1667) >= toro.AND_FLOOR - 0.05
    assert _and_peak(clip, 1.8333) >= toro.AND_FLOOR - 0.05
    assert float(np.max(np.abs(clip))) == 0.9
    # The guard travels with the lead: the same meter-3 "and" is floored when 3
    # leads (turn 2) but stays at its identity level when 3 follows (turn 1).
    assert _and_peak(clip, 1.1667) > 0.10 + _and_peak(clip, 0.333)


def test_each_meter_keeps_its_own_and_identity_in_a_stack():
    # The guard only *adds* a floor underneath the lead's "and": the non-accent
    # meters' upbeats still ring at their own rung (the later ranks), so the
    # per-base identity ladder is untouched in the mix.
    clip = _clip([2, 3, 5])
    assert _and_peak(clip, 0.333) > 0.12    # meter 3's "and" still sounds
    assert _and_peak(clip, 0.6) > 0.12      # meter 5's "and" still sounds
