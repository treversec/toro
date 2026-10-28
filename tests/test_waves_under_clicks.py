"""The "Waves under the clicks" option: each meter's phase-locked wave beneath
its clicks is *audible* — forward into every beat there is a pitched swell that
rings at that beat's *own* pulse frequency (downbeats at the meter pitch, and,
when raised, upbeats at their raised pitch). Each swell peaks at its beat,
sinking to silence at the troughs, so the rise and fall of every wave names
the pulse it belongs to."""
import numpy as np
import pytest
import streamlit as st

import toro


# --- _add_wave (pure helper) ------------------------------------------------

def test_wave_breathes_quietest_at_the_trough_between_beats():
    track = np.zeros(1000)
    toro._add_wave(track, 2, 0.0, 1.0, 40, sr=1000, amp=0.4)  # beats at 0 and 0.5s
    assert track[250] == pytest.approx(0.0, abs=1e-6)  # halfway to the beat: silent


def test_wave_is_loud_near_the_beat():
    track = np.zeros(1000)
    toro._add_wave(track, 2, 0.0, 1.0, 40, sr=1000, amp=0.4)
    assert abs(track[125]) > 0.15  # quarter of the way to the beat: tone swelling up


def test_wave_swells_meet_at_silent_troughs_across_cycles():
    """Each beat's swell fades fully to silence in its trough, so the wave joins
    its next cycle with no seam (each trough reads 0.0) and the constant-pitch
    wave carries no leftover phase across it."""
    w = np.zeros(4000)
    toro._add_wave(w, 3, 0.0, 4.0, 40, sr=1000, amp=0.4)
    assert abs(w[1999] - w[2000]) < 1e-4  # trough between segments: both are silence


def test_wave_pitch_tracks_the_meter():
    track_hi = np.zeros(2000)
    track_lo = np.zeros(2000)
    toro._add_wave(track_hi, 8, 0.0, 2.0, 40, sr=1000, amp=0.4)
    toro._add_wave(track_lo, 2, 0.0, 2.0, 40, sr=1000, amp=0.4)
    # Higher meters wave at a higher pitched tone: around the first butterfly,
    # the 8 sits in a trough while the 2 is up near its crest-quarter.
    mid_beat = int(0.125 * 1000)
    assert abs(track_hi[mid_beat]) < abs(track_lo[mid_beat])


def test_upbeat_wave_pitches_higher_like_the_upbeat_pulse():
    st.session_state["enable_upbeats"] = True
    st.session_state["upbeat_factor"] = 1.5
    track = np.zeros(3000)
    toro._add_wave(track, 3, 0.0, 3.0, 40, sr=1000, amp=0.4)
    # meter 3's beats ring at 0, 1, 2 — beat 1 is the raised upbeat. The swells
    # half-a-beat around beats 1 and 2 share the same outline; the upbeat's
    # crosses zero far more often because its tone is 1.5x the frequency.
    def crossings(x):
        return int(np.sum(np.diff(np.sign(x)) != 0))
    downbeat_swell = track[1600:2400]  # centered on beat 2 (meter pitch, 85 Hz)
    upbeat_swell = track[600:1400]     # centered on beat 1 (raised pitch, 128 Hz)
    # 1.5x the pitch means ~1.5x the zero crossings — a margin, not a coin flip.
    assert crossings(upbeat_swell) > 1.2 * crossings(downbeat_swell)
    st.session_state["enable_upbeats"] = False


# --- render integration (wired through st.session_state) ----------------------

def test_render_meter_track_silent_between_beats_when_disabled():
    st.session_state["waves"] = False
    cycle = toro.render_meter_track(4, 2.0, 40, sr=1000)
    assert abs(cycle[250]) < 0.01  # halfway between beats, clicks long decayed


def test_render_meter_track_adds_wave_between_beats_when_enabled():
    st.session_state["waves"] = True
    cycle = toro.render_meter_track(2, 2.0, 40, sr=1000)
    assert abs(cycle[100]) > 0.1  # swelling quarter-beat region of meter 2, audible


def test_render_node_scales_wave_with_node_amplitude():
    st.session_state["waves"] = True
    track = np.zeros(1000)
    toro.render_node(toro.Node(2), 0.0, 1.0, 40, 0.6, 0.5, track, 1000)
    assert abs(track[250]) == pytest.approx(0.0, abs=1e-6)  # halfway to the 2's beat: silent trough
    track2 = np.zeros(1000)
    toro.render_node(toro.Node(2), 0.0, 1.0, 40, 0.6, 1.0, track2, 1000)
    assert abs(track2[125]) > abs(track[125])  # amp 1.0 (0.2) swells louder than amp 0.5 (0.1)
