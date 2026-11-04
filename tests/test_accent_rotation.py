"""Accent rotation: one clip hands the lead from base to base, turn_len cycles each.

Every test declares the session state it needs (shared st.session_state leaks
across tests, and render_combined reads the waves/upbeat toggles). Renders use
explicit turn/cycle params and short deterministic buffers. The accent's
identity — the lowest pitch rung (rank 1) and its bright crown — is what moves
turn to turn, so the hand-off is proven by those two deterministic signatures.
"""
import numpy as np
import streamlit as st

import toro

SR = 1000


def test_rotation_order_puts_the_radio_pick_first_then_selection_order():
    assert toro.lead_with([5, 3, 2], 3) == [3, 5, 2]     # the accent leads
    assert toro.lead_with([5, 3, 2], 5) == [5, 3, 2]     # first-picked start: unchanged


def test_accent_rotation_length_covers_every_base_for_turn_len_cycles():
    st.session_state["enable_upbeats"] = False
    st.session_state["waves"] = False
    clip = toro.render_rotated([2, 3, 5], 0.5, 40, sr=SR, turn_len=2, accent=5)
    assert len(clip) == 3 * 2 * int(0.5 * SR)     # bases × turn_len × cycle
    clip2 = toro.render_rotated([3], 1.0, 40, sr=SR, turn_len=4, accent=3)
    assert len(clip2) == 4 * SR                    # a single base still yields a clip


def test_accent_rotation_hands_off_loudness_turn_to_turn():
    st.session_state["enable_upbeats"] = False
    st.session_state["waves"] = False
    clip = toro.render_rotated([2, 3], 1.0, 40, sr=SR, turn_len=1, accent=2)

    def dominant(x):
        win = np.hanning(len(x))
        X = np.abs(np.fft.rfft(x * win))
        f = np.fft.rfftfreq(len(x), 1 / SR)
        return f[np.argmax(X)]

    # Turn 1 = [2, 3]: meter 2 leads (rank 1). Turn 2 = [3, 2]: meter 3 leads.
    # The same meter-2 beat grid changes character between turns — the rank-1
    # click rides the lowest rung (40 Hz), the rank-2 click perches higher
    # (66 Hz), and the crown hand-off is proven separately below.
    assert dominant(clip[500:590]) < dominant(clip[1500:1590])   # the lead is lowest
    # And the crown hands off with it: turn 1's beat rings in the crown band,
    # turn 2's (rank 2) does not.
    crowns = (_crown_corr(clip[500:590]), _crown_corr(clip[1500:1590]))
    assert crowns[0] > 2.0 * crowns[1]
    assert np.max(np.abs(clip)) == 0.9              # whole rotation normalized once


def test_rotation_tiles_turns_identically_and_normalizes_with_waves():
    st.session_state["enable_upbeats"] = False
    st.session_state["waves"] = True
    w = toro.render_rotated([2, 3], 1.0, 40, sr=SR, turn_len=2, accent=2)
    assert np.max(np.abs(w)) == 0.9
    # Turn 1 (0–2 s) repeats its own cycle twice, identical for the full two cycles.
    assert w[500] == w[1500]


def _crown_corr(x, sr=SR):
    """How strongly the accent's crown rings in the segment: the coherent
    projection of the window onto a sinusoid at its 4.8× base glint (≈192 Hz at
    base 40). The crown is deterministic, so its projection integrates up;
    the clicks' random crack transients project incoherently and wash out."""
    t = np.arange(len(x)) / sr
    f = 40 * toro.CROWN_RATIO
    return np.hypot(np.sum(x * np.cos(2 * np.pi * f * t)),
                    np.sum(x * np.sin(2 * np.pi * f * t)))


def test_accent_beats_ring_with_a_bright_crown():
    st.session_state["enable_upbeats"] = False
    st.session_state["waves"] = False

    mix = toro.render_meter_mix([2, 3], 1.0, 40, sr=SR)
    # Only the accent (rank 1, meter 2) beats carry the 4.8× crown glint: the
    # full click body of its second beat (~500 ms) rings in the crown band, while
    # meter 3's beat (~333 ms, rank 2) just clicks. The coherent projection is
    # far stronger on the crowned beat — the accent leads low, loud, and glinting.
    crown_beat = _crown_corr(mix[500:590])
    plain_beat = _crown_corr(mix[320:410])
    assert crown_beat > 2.0 * plain_beat


def test_fractal_accent_base_beats_carry_the_crown():
    st.session_state["enable_upbeats"] = False
    st.session_state["waves"] = False

    roots = toro.build_fractal([2, 3], 0, {})
    forest = toro.render_forest(roots, 1.0, 40, sr=SR)
    crown_beat = _crown_corr(forest[500:590])      # rank-1 root's beat glints
    plain_beat = _crown_corr(forest[320:410])      # rank-2 root's beat, no crown
    assert crown_beat > 2.0 * plain_beat
