"""Accent rotation: one clip hands the lead from base to base, turn_len cycles each.

Every test declares the session state it needs (shared st.session_state leaks
across tests, and render_combined reads the waves/upbeat toggles). Renders use
explicit turn/cycle params and short deterministic buffers; the accent hand-off
is proven by loudness (rank 1 = BASIS_GAIN).
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
    # Turn 1 = [2, 3]: meter 2 leads (rank 1, BASIS_GAIN louder). Turn 2 = [3, 2]:
    # meter 3 leads. Compare a meter-2 beat (at an even 0.5s) in each turn.
    seg1_beat = np.max(np.abs(clip[495:505]))    # turn 1's meter-2 beat (rank 1)
    seg2_beat = np.max(np.abs(clip[1495:1505]))  # turn 2's meter-2 beat (rank 2)
    assert seg1_beat > 1.1 * seg2_beat           # the hand-off visibly shifts the lead
    assert np.max(np.abs(clip)) == 0.9           # whole rotation normalized once


def test_rotation_tiles_turns_identically_and_normalizes_with_waves():
    st.session_state["enable_upbeats"] = False
    st.session_state["waves"] = True
    w = toro.render_rotated([2, 3], 1.0, 40, sr=SR, turn_len=2, accent=2)
    assert np.max(np.abs(w)) == 0.9
    # Turn 1 (0–2 s) repeats its own cycle twice, identical for the full two cycles.
    assert w[500] == w[1500]
