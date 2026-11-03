"""Spoken counts: the accent meter says \"one and two and three…\" over the beat grid.

Every test declares the session state it needs (enable_counts / enable_upbeats) —
shared st.session_state leaks across tests otherwise. Mixed with *stub words*
(a short decaying burst per word), never the network TTS.
"""
import numpy as np
import streamlit as st

import toro

SR = 1000


def burst(f, spans=64):
    t = np.arange(spans) / SR
    return (0.8 * np.exp(-6 * t) * np.sin(2 * np.pi * f * t)).astype(np.float32)


def stub_words():
    """One distinct burst per spoken word used by the accent counts."""
    return {"1": burst(220), "2": burst(330), "3": burst(440),
            "4": burst(550), "and": burst(660)}


def window(track, sample, width=20):
    a, b = max(0, sample), min(len(track), sample + width)
    return track[a:b]


def test_counts_off_by_default_silences_the_words():
    st.session_state["enable_counts"] = False
    st.session_state["enable_upbeats"] = True
    track = np.zeros(4000)
    toro.add_counts(track, 3, 1.0, 40, sr=SR, words=stub_words())
    assert np.max(np.abs(track)) < 1e-9  # nothing spoken until the toggle is on


def _word_with_loud_start():
    """A word whose very first sample is loud (a real speech onset), with silent pads."""
    body = np.concatenate([np.full(60, 0.4), burst(220)])
    return np.concatenate([np.zeros(300), body, np.zeros(500)]), body


def test_trim_silence_strips_the_lead_in_so_onsets_are_immediate():
    raw, body = _word_with_loud_start()
    t = toro._trim_silence(raw)
    assert np.allclose(t, body)                  # only the 300-sample lead-in is gone
    assert abs(t[0]) > 0.2                       # sample 0 IS the first audible sample


def test_trim_silence_leaves_tight_words_and_silence_alone():
    tight = _word_with_loud_start()[1]
    t = toro._trim_silence(tight)
    assert len(t) == len(tight) and np.allclose(t, tight)
    assert len(toro._trim_silence(np.zeros(100))) == 0  # all-silence: nothing survives


def test_counts_say_every_beat_and_every_and_of_the_accent():
    st.session_state["enable_counts"] = True
    st.session_state["enable_upbeats"] = True
    track = np.zeros(4000)
    toro.add_counts(track, 3, 1.0, 40, sr=SR, words=stub_words())
    for k in range(3):                       # beats at 0, 333, 666 ms → "one two three"
        assert np.max(np.abs(window(track, int(k * 1000 / 3)))) > 0.2
    for k in range(3):                      # "and" at the midpoint of each gap
        assert np.max(np.abs(window(track, int((k + 0.5) * 1000 / 3)))) > 0.2


def test_counts_follow_the_accent_not_the_first_selection():
    st.session_state["enable_counts"] = True
    st.session_state["enable_upbeats"] = True
    track3 = np.zeros(4000)
    toro.add_counts(track3, 3, 1.0, 40, sr=SR, words=stub_words())
    track2 = np.zeros(4000)
    toro.add_counts(track2, 2, 1.0, 40, sr=SR, words=stub_words())
    # The accent's own second beat lands at 500 ms ("two"); a note at 333 ms only
    # rings for accent 3.
    assert np.max(np.abs(window(track2, 500))) > 0.2      # accent 2 counts "two" at 500
    assert np.max(np.abs(window(track2, 333))) < 0.01     # …but no word at 333
    assert np.max(np.abs(window(track3, 333))) > 0.2      # accent 3's "two" does


def test_counts_respect_the_and_toggle():
    st.session_state["enable_counts"] = True
    st.session_state["enable_upbeats"] = False
    track = np.zeros(4000)
    toro.add_counts(track, 3, 1.0, 40, sr=SR, words=stub_words())
    for k in range(3):
        assert np.max(np.abs(window(track, int(k * 1000 / 3)))) > 0.2   # words still land
    assert np.max(np.abs(window(track, 500))) < 0.01    # midpoints stay wordless
