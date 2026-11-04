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
    st.session_state["enable_upbeats"] = False
    track = np.zeros(1000)
    toro._add_wave(track, 2, 0.0, 1.0, 40, sr=1000, amp=0.4)  # beats at 0 and 0.5s
    assert track[250] == pytest.approx(0.0, abs=1e-6)  # halfway to the beat: silent


def test_wave_is_loud_near_the_beat():
    st.session_state["enable_upbeats"] = False
    track = np.zeros(1000)
    toro._add_wave(track, 2, 0.0, 1.0, 40, sr=1000, amp=0.4)
    assert abs(track[125]) > 0.15  # quarter of the way to the beat: tone swelling up


def test_wave_swells_meet_at_silent_troughs_across_cycles():
    """Each beat's swell fades fully to silence in its trough, so the wave joins
    its next cycle with no seam (each trough reads 0.0) and the constant-pitch
    wave carries no leftover phase across it."""
    st.session_state["enable_upbeats"] = False
    w = np.zeros(4000)
    toro._add_wave(w, 3, 0.0, 4.0, 40, sr=1000, amp=0.4)
    assert abs(w[1999] - w[2000]) < 1e-4  # trough between segments: both are silence


def test_wave_pitch_tracks_the_meter():
    st.session_state["enable_upbeats"] = False
    track_hi = np.zeros(2000)
    track_lo = np.zeros(2000)
    toro._add_wave(track_hi, 8, 0.0, 2.0, 40, sr=1000, amp=0.4)
    toro._add_wave(track_lo, 2, 0.0, 2.0, 40, sr=1000, amp=0.4)
    # Higher meters wave at a higher pitched tone: around the first butterfly,
    # the 8 sits in a trough while the 2 is up near its crest-quarter.
    mid_beat = int(0.125 * 1000)
    assert abs(track_hi[mid_beat]) < abs(track_lo[mid_beat])


def test_upbeat_pitch_factor_defaults_to_two():
    st.session_state.pop("upbeat_factor", None)   # fresh default — not a leftover factor
    # The accent rung (rank 1) sits at the slider's factor: exactly ×2.0.
    assert toro.and_freq(3, 40, rank=1) == pytest.approx(toro.meter_freq(3, 40, rank=1) * 2.0)


def test_and_pitch_ladder_climbs_with_selection_rank():
    st.session_state["upbeat_factor"] = 2.0
    # The and-beat ratio is the and-ladder itself: the accent rings 2.0× its beat,
    # and each later rank rings a little higher on top of that.
    r1 = toro.and_freq(5, 40, rank=1) / toro.beat_freq(5, 40, 0, rank=1)
    r2 = toro.and_freq(5, 40, rank=2) / toro.beat_freq(5, 40, 0, rank=2)
    r3 = toro.and_freq(5, 40, rank=3) / toro.beat_freq(5, 40, 0, rank=3)
    assert r1 == pytest.approx(2.0)                 # the slider's factor, undiluted
    assert r2 > 1.1 * r1 and r3 > 1.1 * r2          # the ladder strictly climbs
    solo = toro.and_freq(3, 40) / toro.beat_freq(3, 40, 0)   # rank omitted → own rung
    assert solo > r1                                # meter 3's ands ride above the accent's
    assert toro.and_freq(3, 40) > toro.and_freq(3, 40, rank=1)


def test_and_brush_is_a_light_high_pitched_swish():
    # The "and" of jazz brushes: LIGHTER than its beat click, and a clearly
    # PITCHED tone at its own (higher) rung — the tone carries a soft high-band
    # brush air, and the whole stroke peaks at 0.55. Pitch reads by spectrum:
    # the stroke's dominant tone sits on its rung, and doubling the rung doubles
    # the dominant frequency — so the wave's low beat → high "and" glide reads.
    sr = 44100
    lo = toro.make_and_click(240, sr=sr, meter=3)
    hi = toro.make_and_click(480, sr=sr, meter=3)

    def dominant(x):
        win = np.hanning(len(x))
        X = np.abs(np.fft.rfft(x * win))
        f = np.fft.rfftfreq(len(x), 1 / sr)
        return f[np.argmax(X)]

    assert np.max(np.abs(hi)) == pytest.approx(0.55, abs=1e-3)   # lighter: soft brush
    beat = toro.make_click(120, sr=sr)
    assert np.max(np.abs(lo)) < np.max(np.abs(beat))             # under its open click
    assert dominant(hi) > 1.8 * dominant(lo)                    # pitched: 2× the rung, 2× the tone
    assert dominant(lo) == pytest.approx(240, rel=0.1)          # and not adrift toward noise


def test_and_tick_brightness_differs_by_meter():
    st.session_state["enable_upbeats"] = True
    sr = 44100

    def high_band_ratio(x):
        win = np.hanning(len(x))
        X = np.abs(np.fft.rfft(x * win))
        f = np.fft.rfftfreq(len(x), 1 / sr)
        return np.sum(X[f >= 2500] ** 2) / np.sum(X ** 2)

    tick2 = toro.make_and_click(124, sr=sr, meter=2)
    tick5 = toro.make_and_click(124, sr=sr, meter=5)
    tick9 = toro.make_and_click(124, sr=sr, meter=9)
    # Higher meters' brush air rides higher (drier/crisper), so the same short
    # stroke holds a larger share of energy in the top band — the meter's own
    # "and" shade — while the stroke stays short, closed, and deterministic.
    assert high_band_ratio(tick9) > 1.5 * high_band_ratio(tick2)
    assert high_band_ratio(tick5) > high_band_ratio(tick2)        # the climb is monotonic
    assert np.allclose(tick5, toro.make_and_click(124, sr=sr, meter=5))  # deterministic


def test_and_rings_at_the_upbeat_factor():
    st.session_state["enable_upbeats"] = True
    st.session_state["upbeat_factor"] = 1.5
    # The "and" between the beats rings at the upbeat pitch — the pulse pitch times
    # the factor (which climbs a ladder by selection rank, meter 3 on its own rung),
    # while every beat itself rings at the pulse's own pitch.
    assert toro.and_freq(3, 40) == pytest.approx(toro.meter_freq(3, 40) * 1.5 * (1 + 0.12 * 2))
    assert toro.beat_freq(3, 40, 1) == toro.beat_freq(3, 40, 0)
    st.session_state["enable_upbeats"] = False


def test_and_sound_is_closed_staccato():
    st.session_state["enable_upbeats"] = True
    a = toro.make_and_click(124, sr=1000)              # the "and": closed, no ring
    tail = np.abs(a[int(len(a) * 0.6):])
    assert len(a) < 0.06 * 1000                         # short, staccato
    assert np.max(tail) < 0.15 * np.max(np.abs(a))      # and the tail is a closed death
    open_click = toro.make_click(124, sr=1000)          # beats keep the open ring
    open_tail = np.abs(open_click[int(len(open_click) * 0.6):])
    assert np.max(open_tail) > 0.15 * np.max(np.abs(open_click))


def test_and_click_plays_between_the_beats():
    st.session_state["enable_upbeats"] = True
    st.session_state["waves"] = False
    track = toro.render_meter_track(2, 1.0, 40, sr=1000)  # beats at 0 & 0.5s
    assert max(abs(track[249:253])) > 0.2  # the "and" sounds at 0.25s — "one *and*, two"
    st.session_state["enable_upbeats"] = False
    quiet = toro.render_meter_track(2, 1.0, 40, sr=1000)
    assert abs(quiet[250]) < 0.01    # toggle off: nothing between the beats


def test_wave_peaks_at_the_and_and_glides_up():
    st.session_state["enable_upbeats"] = True
    st.session_state["upbeat_factor"] = 1.5
    t = np.zeros(1000)
    toro._add_wave(t, 2, 0.0, 1.0, 40, sr=1000, amp=1.0)  # beats at 0 & 0.5, "and" at 0.25
    def crossings(x):
        return int(np.sum(np.diff(np.sign(x)) != 0))
    # The swell crests on the "and" and sinks to silence at the beats, and its tone
    # glides from the downbeat pitch up to the upbeat pitch, so the dense stretch
    # around the "and" crosses zero far more often than the sparse stretch right
    # after a beat — a margin, not a coin flip.
    assert max(abs(t[200:300])) > 0.5   # the swell's peak sits on the "and"
    assert abs(t[480]) < 0.1            # and fades to silence at the beats
    assert crossings(t[200:300]) > 1.2 * crossings(t[0:100])
    st.session_state["enable_upbeats"] = False


def test_render_node_plays_an_and_between_each_beat():
    st.session_state["enable_upbeats"] = True
    st.session_state["waves"] = True
    track = np.zeros(3000)
    toro.render_node(toro.Node(3), 0.0, 3.0, 40, 0.6, 1.0, track, 1000)
    assert max(abs(track[490:510])) > 0.15  # the "and" sounds at 0.5s, between beat 0 and beat 1
    st.session_state["enable_upbeats"] = False


def test_render_node_keys_pitch_by_selection_order():
    st.session_state["enable_upbeats"] = False
    st.session_state["waves"] = True
    lo = np.zeros(3000)
    toro.render_node(toro.Node(3, rank=1), 0.0, 3.0, 40, 0.6, 1.0, lo, 1000)
    hi = np.zeros(3000)
    toro.render_node(toro.Node(3, rank=3), 0.0, 3.0, 40, 0.6, 1.0, hi, 1000)
    def crossings(x):
        return int(np.sum(np.diff(np.sign(x)) != 0))
    # The recursed sub-patterns share the same selection order at every depth, so
    # a node at a higher rank rings at a higher rung of the pitch ladder.
    assert crossings(hi[1200:1800]) > 1.2 * crossings(lo[1200:1800])

def test_build_fractal_ranks_by_selection_order_at_every_depth():
    roots = toro.build_fractal([3, 2], 2, {3: {0}, 2: {0}})
    def walk(node, out):
        out.append((node.meter, node.rank))
        for p in node.patterns:
            walk(p, out)
    pairs = []
    for root in roots:
        walk(root, pairs)
    # "Order at every depth": the base picked first (3) is the foundation (rank
    # 1) at the root *and* inside every recursion — the later pick (2) always
    # perches one rung higher, never the meter's own number.
    assert set(pairs) == {(3, 1), (2, 2)}

def test_render_meter_track_silent_between_beats_when_disabled():
    st.session_state["enable_upbeats"] = False
    st.session_state["waves"] = False
    cycle = toro.render_meter_track(4, 2.0, 40, sr=1000)
    assert abs(cycle[250]) < 0.01  # halfway between beats, clicks long decayed


def test_render_meter_track_adds_wave_between_beats_when_enabled():
    st.session_state["enable_upbeats"] = False
    st.session_state["waves"] = True
    cycle = toro.render_meter_track(2, 2.0, 40, sr=1000)
    assert abs(cycle[100]) > 0.1  # swelling quarter-beat region of meter 2, audible


def test_render_node_scales_wave_with_node_amplitude():
    st.session_state["enable_upbeats"] = False
    st.session_state["waves"] = True
    track = np.zeros(1000)
    toro.render_node(toro.Node(2), 0.0, 1.0, 40, 0.6, 0.5, track, 1000)
    assert abs(track[250]) == pytest.approx(0.0, abs=1e-6)  # halfway to the 2's beat: silent trough
    track2 = np.zeros(1000)
    toro.render_node(toro.Node(2), 0.0, 1.0, 40, 0.6, 1.0, track2, 1000)
    assert abs(track2[125]) > abs(track[125])  # amp 1.0 (0.2) swells louder than amp 0.5 (0.1)


# --- selection order = the perspective ----------------------------------------

def test_rank_keys_pitch_higher_for_later_selections():
    st.session_state["enable_upbeats"] = False
    low = np.zeros(3000)
    toro._add_wave(low, 3, 0.0, 3.0, 40, rank=1, sr=1000, amp=0.4)
    high = np.zeros(3000)
    toro._add_wave(high, 3, 0.0, 3.0, 40, rank=3, sr=1000, amp=0.4)
    def crossings(x):
        return int(np.sum(np.diff(np.sign(x)) != 0))
    # A later rank means a higher rung on the pitch ladder: the rank-3 wave crosses
    # clearly more tone-cycles per swell than the rank-1 wave over the same outline.
    assert crossings(high[1200:1800]) > 1.2 * crossings(low[1200:1800])


def test_lead_with_puts_the_accent_first_keeping_selection_order():
    assert toro.lead_with([5, 3, 2], 3) == [3, 5, 2]
    assert toro.lead_with([5, 3, 2], 5) == [5, 3, 2]  # first-picked accent: unchanged
    assert toro.lead_with([2], 2) == [2]


def test_first_selection_sounds_louder_than_the_same_meter_second():
    st.session_state["enable_upbeats"] = False
    st.session_state["waves"] = True
    first = toro.render_meter_mix([5, 3], 2.0, 40, sr=1000)   # 5 entered first: the foundation
    second = toro.render_meter_mix([3, 5], 2.0, 40, sr=1000)  # 5 entered second
    i = int(1.9 * 1000)  # a sample away from every click, so this is wave-only
    assert abs(first[i]) > abs(second[i])
