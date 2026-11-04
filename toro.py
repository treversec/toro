import io
import wave

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

SR = 44100

# ---------------------------------------------------------------------------
# Audio synthesis
# ---------------------------------------------------------------------------
# Toro's "harmonic perspective of rhythm" treats each simultaneous meter as a
# different frequency of the same fundamental pulse -- the same way overtones
# relate in a harmonic series. We take that literally: meter "n" is rendered
# both as n evenly-spaced clicks per cycle AND as a click pitched at the nth
# harmonic of a chosen fundamental. Hearing it and seeing it line up is the
# point.

def _make_crown(base_freq, sr=SR, dur=0.06):
    """The accent's crown: a bright high glint at CROWN_RATIO × the base pitch —
    deterministic (no noise source), so every accent beat rings the same clear
    overtone, short enough to read as a glint on the beat rather than a ring."""
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    f = base_freq * CROWN_RATIO
    attack = 1.0 - np.exp(-250 * t)
    glint = np.sin(2 * np.pi * f * t) * attack * np.exp(-40 * t)
    glint += 0.3 * np.sin(2 * np.pi * 2.74 * f * t) * attack * np.exp(-60 * t)
    peak = np.max(np.abs(glint))
    return glint / peak if peak > 0 else glint


def _click_beep(freq, sr, dur, decay):
    """The original sound: a short pitched beep."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t)
    transient = np.random.uniform(-1, 1, len(t)) * np.exp(-90 * t) * 0.35
    return (tone + transient) * np.exp(-decay * t)


def _click_drum(freq, sr, dur, decay):
    """A synthesized drum hit, still pitched at `freq`: a fast downward pitch
    sweep on the attack gives the "thump", a bright noise burst gives the
    initial "crack", and a punchy exponential decay keeps it percussive."""
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    sweep = freq * (1.0 + 1.9 * np.exp(-40 * t))
    phase = 2 * np.pi * np.cumsum(sweep) / sr
    body = np.sin(phase) * np.exp(-decay * t)
    crack = np.random.uniform(-1, 1, n) * np.exp(-180 * t) * 0.5
    return body + crack


_SOUNDS = {"Pitched beeps (original)": _click_beep,
           "Drums (synthesized)": _click_drum}

def make_click(freq, sr=SR, dur=0.09, decay=22):
    """Generate one note using the currently selected sound.

    `st.session_state["sound"]` holds the chosen sound name (see `_SOUNDS`);
    falls back to the pitched beeps if unset. The chosen generator keeps the
    `freq` argument, so meter n stays pitched at its mapped frequency either
    way.
    """
    name = st.session_state.get("sound", "Pitched beeps (original)")
    return _SOUNDS.get(name, _click_beep)(freq, sr=sr, dur=dur, decay=decay)


def make_and_click(freq, sr=SR, dur=0.045, meter=1):
    """The "and": a light brush stroke in the jazz sense — short and closed, but
    *pitched*: a clean tone at the and's own rung (already ×2.0+ its beat and
    climbing the ladder) dominates, so every "and" is clearly a higher note than
    its click and the wave's low beat → high "and" oscillation reads easily.
    A soft deterministic high-band brush air rides underneath (~30%) for the
    brushes softness; `meter` shades the air's band (drier, crisper, higher for
    bigger meter numbers), so each base's "and" keeps its own voice without
    ever needing a palette that could run out. No noise source, so the brush is
    identical every play; the whole stroke is normalized to a light 0.55 peak.
    """
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    tone = np.sin(2 * np.pi * freq * t) * \
        (1.0 - np.exp(-250 * t)) * np.exp(-(100 + 8 * (meter - 1)) * t)  # pitched body
    f0 = 2000 + 150 * meter          # the brush air's rising hand-off
    f1 = 3200 + 200 * meter
    air = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t ** 2 / (2 * dur))) * \
        np.exp(-150 * t)
    out = tone + 0.3 * air
    peak = np.max(np.abs(out))
    return out / peak * 0.55 if peak > 0 else out


def _accent_deep_drum(freq, sr=SR, dur=0.16):
    """A clean, sub-heavy kick for the start of the cycle. A short downward
    pitch sweep gives the "thump"; the decay is kept fast and tight so it reads
    as a single downbeat rather than a ringing boom (no echo)."""
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    sweep = freq * (1.0 + 1.6 * np.exp(-45 * t))
    phase = 2 * np.pi * np.cumsum(sweep) / sr
    body = np.sin(phase) * np.exp(-28 * t)
    return body * 0.9


def _accent_high_chime(freq, sr=SR, dur=0.35):
    """A bright, bell-like chime for the start of the cycle: a sine with
    inharmonic partials and a slow decay, so it rings out clearly over the
    metered clicks."""
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    env = np.exp(-9 * t)
    partials = (1.0 * np.sin(2 * np.pi * freq * t)
                + 0.5 * np.sin(2 * np.pi * freq * 2.76 * t)
                + 0.3 * np.sin(2 * np.pi * freq * 5.4 * t))
    return partials * env


_CYCLE_ACCENTS = {"Deep drum": _accent_deep_drum,
                  "High chime": _accent_high_chime}


def make_cycle_accent(sr=SR):
    """The indicator sound for the start of every cycle (where all meters'
    beats intersect), selected from `st.session_state["cycle_accent"]`; returns
    a zero-length array when set to "Off" or unset. The pitch is anchored low
    so the deep drum/ringing chime reads as a clear downbeat alongside the
    (louder) metered voices."""
    name = st.session_state.get("cycle_accent", "Off")
    gen = _CYCLE_ACCENTS.get(name, None)
    if gen is None:
        return np.zeros(0)
    return gen(80.0, sr=sr)


def add_cycle_accent(track, sr=SR):
    """Add the cycle-start indicator at t=0 if one is selected. `track` is
    modified in place; the indicator is added just before each cycle's
    downbeat so it marks where every meter's first beat intersects."""
    accent = make_cycle_accent(sr=sr)
    if len(accent) == 0:
        return
    add_wrapped(track, accent, 0, sr=sr)


def add_wrapped(track, click, start_idx, sr=SR):
    """Add a click to a loop-safe track at start_idx, wrapping any overflow
    back to the start so the waveform loops seamlessly."""
    total = len(track)
    idx = int(start_idx)
    end = min(idx + len(click), total)
    if end > idx:
        track[idx:end] += click[: end - idx]
    if end < idx + len(click):
        overflow = click[end - idx:]
        burst = min(len(overflow), idx)
        track[:burst] += overflow[:burst]
    return track


def _add_wave(track, meter, start, end, base_freq, sr=SR, amp=1.0, rank=None):
    """Mix the audible phase-locked waves under the clicks, one swell per beat,
    each pitched at that beat's *own* pulse frequency … every swell rises from
    silence in the trough after the previous click, sinks to silence at the next,
    and the curve swallows the groove. `amp` scales how loud the waves are;
    fractal layers pass their own amplitude down, so deeper layers swell quieter,
    exactly like their clicks. `rank` (position in the selection order) keys the
    pitch; when omitted the meter's own number does.

    When "Play 'and' between each beat" is on, each swell *crests on the "and"*
    — halfway between two beats — and its tone glides from the downbeat pitch
    up to the upbeat pitch (`upbeat_factor` ×) as it swells, so the rise and
    fall of every wave names the pulse that leads into its beat.
    """
    span = end - start
    n = len(track)
    beat_sr = span / meter * sr           # samples from one trough to the next
    width = int(beat_sr)
    half = beat_sr / (2 * sr)             # seconds from a beat to its troughs
    if st.session_state.get("enable_upbeats", DEFAULT_UPBEATS):
        period = span / meter             # one swell per beat-interval
        f0 = meter_freq(meter, base_freq, rank)
        f_and = and_freq(meter, base_freq, rank)
        for k in range(meter):
            tb = start + k * period
            t = np.linspace(tb, tb + period, width, endpoint=False)
            env = (1.0 + np.cos(2 * np.pi * (t - tb - half) / period)) / 2.0
            freq = f0 + (f_and - f0) * env
            tone = np.sin(2 * np.pi * np.cumsum(freq) / sr)
            seg = amp * tone * env
            a = int(tb * sr)
            i1 = min(a + width, n)
            track[a:i1] += seg[:i1 - a]
            if a + width > n:
                track[: a + width - n] += seg[i1 - a:]
        return track
    for k in range(meter):
        tb = start + k * span / meter     # the click this swell leads into
        center = int(tb * sr)            # the click's own sample
        t = np.linspace(tb - half, tb + half, width, endpoint=False)
        env = (1.0 + np.cos(2 * np.pi * (t - tb) * sr / beat_sr)) / 2.0
        tone = np.sin(2 * np.pi * beat_freq(meter, base_freq, k, rank) * (t - tb))
        seg = amp * tone * env
        a = (center - width // 2) % n
        i1 = min(a + width, n)
        track[a:i1] += seg[:i1 - a]
        if a + width > n:
            track[: a + width - n] += seg[i1 - a:]
    return track


def meter_freq(n, base_freq, rank=None):
    """Map a meter — or a *position in the selection order* — to its drum
    frequency, ascending gently with compression. `rank` is the meter's position
    in the selection order (1 = the first-picked foundation), so the pitch
    follows the user's chosen perspective; when omitted, the meter's own number
    keys the pitch.

    Strict harmonics (`base_freq * n`) leap a full overtone per meter and climb
    too fast for low drums. Instead, low meters stay close together and the
    climb softens as we ascend, so the base of the pyramid sits low while
    keeping a rising, still-distinct pitch per meter.
    """
    if rank is not None:
        n = rank
    return base_freq * (1 + 0.65 * (n - 1) ** 0.8)


# Defaults for the "Play 'and' between each beat" option (see sidebar widgets).
# Session state holds the live values: enable_upbeats (bool) and upbeat_factor (float).
DEFAULT_UPBEATS = True   # the "and"s sound by default — turn them off in the sidebar
DEFAULT_UPBEAT_FACTOR = 2.0


def beat_freq(n, base_freq, beat_index, rank=None):
    """Pitch of one beat of a meter: every beat rings at the meter's own pitch —
    the (rank-keyed) position's rung, so each pulse leads with its own voice.
    When "Play 'and' between each beat" is on, the between-beat "and" rings
    `upbeat_factor` higher (`and_freq`); `beat_index` is kept so call sites say
    which beat they ring, even though beats now share one pitch.
    """
    return meter_freq(n, base_freq, rank)


def and_freq(n, base_freq, rank=None):
    """Pitch of the "and" that plays halfway between a meter's beats: the meter's
    own pitch times the `upbeat_factor` — climbing a *ladder* by selection rank
    (each later rank sits 12% higher than the last, on top of the factor), so
    upbeats of neighboring bases are separated by pitch as well as by transience.
    Falls back to the meter's own number as its rung when no `rank` is given.
    Reads the factor from `st.session_state`, so it's configurable from the
    sidebar.
    """
    factor = st.session_state.get("upbeat_factor", DEFAULT_UPBEAT_FACTOR)
    rung = rank if rank is not None else n
    return meter_freq(n, base_freq, rank) * factor * (1 + 0.12 * (rung - 1))


# The "and" between beats is a light high brush — its stroke is normalized to a
# soft 0.55 peak and stays pitched at its (higher) rung, so the subdivision
# reads as "one *and*, two *and*" — lighter, higher — under the beats.
AND_GAIN = 0.55

# Spoken counts are off until toggled on (see sidebar widgets). Session state
# holds the live value: enable_counts (bool).
DEFAULT_COUNTS = False


def _trim_silence(a, thresh=1e-3):
    """Cut the silent pre-roll and tail off a spoken word, leaving the voice
    itself untouched. What survives starts at its first audible sample — so a
    count word placed at a beat sample *starts* on that beat rather than after
    the engine's lead-in pause."""
    nz = np.nonzero(np.abs(a) > thresh)[0]
    if len(nz) == 0:
        return a[:0]
    return a[nz[0]:nz[-1] + 1]


@st.cache_data(show_spinner=False)
def _spoken_word(word):
    """Speak one count word ("1", "and", …) into float samples at the app's SR,
    trimmed so the first sample is the word's actual onset. edge-tts (pure
    Python, Microsoft's Edge voice) both speaks faster than gTTS — the counts
    come out staccato — and returns 44.1 kHz audio natively, plus a fast rate
    ("+50%") keeps each word to about a third of a second. Each word costs one
    short network call, once, cached per word.
    """
    import edge_tts, miniaudio
    chunks = b"".join(c["data"] for c in
                      edge_tts.Communicate(word, rate="+50%").stream_sync()
                      if c["type"] == "audio")
    dec = miniaudio.decode(chunks, output_format=miniaudio.SampleFormat.FLOAT32)
    return _trim_silence(np.array(dec.samples, dtype=np.float32))


def add_counts(track, accent, cycle_len, base_freq, sr=SR, words=None):
    """Spoken counts for the accent meter: the accent says its numbers — "1", "2",
    … with each word starting on one of its beats, and an "and" at each and — so
    the lead meter's count rides over the rhythm. `words` is a dict of
    word→samples (tests pass stubs and never touch the network); the app lets
    it default to the cached gTTS voice. Nothing is added while the spoken-
    counts toggle is off.
    """
    if not st.session_state.get("enable_counts", DEFAULT_COUNTS):
        return track

    def audio(w):
        return _spoken_word(w) if words is None else words[w]

    period = cycle_len / accent
    for k in range(accent):
        add_wrapped(track, audio(str(k + 1)), int(k * period * sr))
        if st.session_state.get("enable_upbeats", DEFAULT_UPBEATS):
            add_wrapped(track, audio("and"), int((k + 0.5) * period * sr))
    return track


def render_meter_track(n, cycle_len, base_freq, sr=SR, rank=None,
                       delay_upbeats=False):
    """n evenly spaced clicks across one cycle, pitched by its *position in the
    selection order*: rank 1 (the first-picked meter) is the foundation and rings
    the lowest pitch, rank 2 the next, and so on — so the user's selection order
    sets the perspective. Calls without a `rank` keep the meter's own number.

    Downbeats ring at the (rank-keyed) base pitch, while upbeats (the even
    2nd/4th/6th...) ring a raised pitch, so the two are easy to tell apart.
    `delay_upbeats` withholds this meter's upbeats for the accent-meter callers
    that re-stamp them through `add_lead_and_guard` (the loudness floor); the
    meter's other content — beats, crown, wave crest on the upbeat — still
    sounds exactly as before.

    Returns a single cycle built to loop seamlessly: a click that would
    overhang the end of the cycle is wrapped back onto the start, so the
    waveform is continuous across the loop point and the underlying pulse
    is never broken by a seam or pause.
    """
    total = int(cycle_len * sr)
    track = np.zeros(total)
    positions = np.linspace(0, cycle_len, n, endpoint=False)
    for i, p in enumerate(positions):
        click = make_click(beat_freq(n, base_freq, i, rank), sr=sr)
        add_wrapped(track, click, int(p * sr))
        if rank == 1:  # the foundation's beats wear the bright crown glint
            crown = _make_crown(base_freq, sr=sr) * CROWN_GAIN
            add_wrapped(track, crown, int(p * sr))
        if st.session_state.get("enable_upbeats", DEFAULT_UPBEATS) and not delay_upbeats:
            and_p = p + cycle_len / n / 2  # the "and" between this beat and the next
            add_wrapped(track, make_and_click(and_freq(n, base_freq, rank), sr=sr,
                                              meter=n) * AND_GAIN,
                         int(and_p * sr))
    if st.session_state.get("waves", False):
        wave_amp = st.session_state.get("wave_amp", DEFAULT_WAVE_AMP)
        _add_wave(track, n, 0.0, cycle_len, base_freq, sr=sr, amp=wave_amp, rank=rank)
    return track


# The accent meter's upbeats carry a *loudness floor*: though each downbeat's
# "and" is mixed at AND_GAIN, the final clip peak (beats + crowns + waves) grows
# as bases stack and the 0.9 normalization would crush it — so the accent's
# upbeats are re-stamped at AND_FLOOR × the mix peak before the normalize. They
# land at AND_FLOOR in the finished clip (the same fullness as solo), still
# lighter than the 0.9 click, while other meters keep their identity "and" level.
AND_FLOOR = 0.33

# Default for the "waves under the clicks" option (see sidebar widgets).
# Session state holds: waves (bool) and wave_amp (float).
DEFAULT_WAVE_AMP = 0.4

# The first-picked meter is the *foundation*: as well as ringing the lowest
# pitch, it is mixed slightly louder so it leads the polyrhythm rather than
# competing with it.
BASIS_GAIN = 1.35

# The accent also wears a *crown*: each of its beats gains a bright high glint at
# CROWN_RATIO × the base pitch (≈265 Hz at base 55, clearing the stack's top of
# ~4.4× at any meter count), mixed at CROWN_GAIN, on top of the low+loud body —
# so the lead reads "low + loud + glinting" and rotation hand-offs visibly move.
CROWN_RATIO = 4.8
CROWN_GAIN = 0.5


def lead_with(chosen, accent):
    """Order a perspective: the accent meter first (the foundation — lowest rung,
    slightly louder), the rest keeping their selection order after it."""
    return [accent] + [m for m in chosen if m != accent]


def add_lead_and_guard(cycle, accent, cycle_len, base_freq, sr=SR):
    """Re-stamp the accent meter's upbeats at AND_FLOOR × the mix peak so they
    outlive the final 0.9 normalize — the stacked beats/crowns/waves would
    otherwise use the whole peak budget up and crush the "and". Called with the
    accent's own upbeats withheld (`delay_upbeats`), before normalization.

    Each swipe is aligned with whatever the mix already holds at its own peak
    sample (phase-flipped to reinforce it, the way a wave crest carries the
    "and"), so the composite reaches AND_FLOOR *whatever the background happens
    to be doing* — a fixed-level add could otherwise be buried by an
    anti-phased swell. The result still reads lighter than the 0.9 click, and
    other meters' identity upbeats keep AND_GAIN. Wrapped like every upbeat, so
    the loop point stays seamless."""
    if not st.session_state.get("enable_upbeats", DEFAULT_UPBEATS):
        return cycle
    peak = np.max(np.abs(cycle))
    if peak <= 0:
        return cycle
    target = peak * AND_FLOOR / 0.9
    total = len(cycle)
    for p in np.linspace(0, cycle_len, accent, endpoint=False):
        click = make_and_click(and_freq(accent, base_freq, 1), sr=sr, meter=accent)
        span = np.max(np.abs(click))
        if span <= 0:
            continue
        click = click / span                  # unit peak: its loudest sample reads 1
        k = int(np.argmax(np.abs(click)))
        mid = int((p + cycle_len / accent / 2) * sr)
        bg = cycle[(mid + k) % total]         # the mix exactly under that sample
        amp = max(0.6, target - abs(bg))      # lift to the floor; keep the body's shape
        if bg * click[k] < 0:
            click = -click                    # reinforce the background, don't fight it
        add_wrapped(cycle, click * amp, mid)
    return cycle


def render_meter_mix(meters, cycle_len, base_freq, sr=SR):
    """Mix the given meters into one (mono) cycle, adding the cycle accent. The
    first-picked meter is the foundation: it keeps the lowest pitch and a slight
    loudness boost (BASIS_GAIN), so it leads the polyrhythm."""
    cycle = np.zeros(int(cycle_len * sr))
    for idx, n in enumerate(meters):
        gain = BASIS_GAIN if idx == 0 else 1.0
        cycle += render_meter_track(n, cycle_len, base_freq, sr=sr, rank=idx + 1,
                                    delay_upbeats=(idx == 0)) * gain
    add_cycle_accent(cycle, sr=sr)
    if meters:  # the lead's upbeats get their loudness floor before normalize
        add_lead_and_guard(cycle, meters[0], cycle_len, base_freq, sr=sr)
    return cycle


def _normalize(cycle):
    peak = np.max(np.abs(cycle))
    if peak > 0:
        cycle = cycle / peak * 0.9
    return cycle


def render_combined(meters, cycle_len, base_freq, sr=SR, count_words=None):
    """Render the base pyramid into one cycle.

    The selection *order* is the perspective: `meters` is used in the order it is
    given, each position setting that meter's rank — the first-picked meter is the
    foundation (lowest pitch), the next the one after, and so on.

    Returns a mono array by default. When `st.session_state["stereo"]` is set,
    the meters are split across two channels (half on the left, half on the
    right) and the result is a stereo array of shape `(2, N)`; the cycle-start
    indicator stays centred on both channels.

    When the spoken-counts toggle is on, the accent meter (`meters[0]`, the
    foundation) says its numbers over its own beat grid, centred on both
    channels like the cycle-start indicator. `count_words` is a word→samples
    dict used by tests; the app lets it default to the cached gTTS voice.
    """
    if not st.session_state.get("stereo", False):
        cycle = render_meter_mix(meters, cycle_len, base_freq, sr=sr)
        if meters and st.session_state.get("enable_counts", DEFAULT_COUNTS):
            add_counts(cycle, meters[0], cycle_len, base_freq, sr=sr, words=count_words)
        return _normalize(cycle)

    ordered = list(meters)
    if len(ordered) < 2:
        left = render_meter_mix(ordered, cycle_len, base_freq, sr=sr)
        if ordered and st.session_state.get("enable_counts", DEFAULT_COUNTS):
            add_counts(left, ordered[0], cycle_len, base_freq, sr=sr, words=count_words)
        return np.vstack([left, left])
    mid = (len(ordered) + 1) // 2
    left = render_meter_mix(ordered[:mid], cycle_len, base_freq, sr=sr)
    right = render_meter_mix(ordered[mid:], cycle_len, base_freq, sr=sr)
    if st.session_state.get("enable_counts", DEFAULT_COUNTS):
        add_counts(left, ordered[0], cycle_len, base_freq, sr=sr, words=count_words)
        add_counts(right, ordered[0], cycle_len, base_freq, sr=sr, words=count_words)
    stereo = np.vstack([left, right])
    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo = stereo / peak * 0.9
    return stereo


def _finish_rotation(parts, turn_len):
    """Concatenate one-cycle segments, `turn_len` copies of each, into one
    normalized clip. Mono and stereo (2, N) buffers both concatenate along the
    last axis, and the whole clip is peaked at 0.9 once."""
    clip = np.concatenate([np.concatenate([p] * turn_len, axis=-1) for p in parts], axis=-1)
    return _normalize(clip)


def render_rotated(meters, cycle_len, base_freq, sr=SR, turn_len=2, accent=None,
                   count_words=None):
    """Rotate the accent through every selected meter inside one clip: the
    radio's pick leads first, then each following base in selection order, with
    `turn_len` cycles at each hand-off. The minimum clip covers every base once
    (bases × turn_len cycles); the whole rotation peaks at 0.9 like a single
    cycle. Spoken counts (when toggled on) hand off with the accent, because
    each turn is rendered through `render_combined` with that turn's ordering.
    """
    if not meters:
        return np.zeros(0)
    order = lead_with(meters, meters[0] if accent is None else accent)
    parts = [
        render_combined(lead_with(meters, m), cycle_len, base_freq, sr=sr,
                        count_words=count_words)
        for m in order
    ]
    return _finish_rotation(parts, turn_len)


# ---------------------------------------------------------------------------
# Rhythm fractals (patterns layered on patterns)
# ---------------------------------------------------------------------------
# A fractal is a rhythm-tree: every node is a meter that sounds on its own,
# and it may carry a *sub-pattern* that plays over every one of its beats.
# So "3/2 on top of the 2" is a 2-base with a 3/2 pattern (a 3-base carrying a
# 2) layered over each of its beats. Every level's own meter always sounds;
# sub-patterns are quieter; pitch follows the base selection order (first base
# picked = foundation) at every nesting depth.

class Node:
    """One pattern: `meter` clicks, plus a *combination* of sub-patterns played
    over some of those clicks (arbitrarily deep). `on` is the set of beat
    indices (0-based) whose pulse carries the sub-patterns; empty `on` or empty
    `patterns` means "no sub-layer on any beat"."""
    def __init__(self, meter, patterns=None, on=None, rank=None):
        self.meter = meter
        self.rank = rank
        self.patterns = patterns if patterns is not None else []
        self.on = set(on) if on is not None else set()

    def depth(self):
        if not self.patterns or not self.on:
            return 0
        return 1 + max(p.depth() for p in self.patterns)

    def __repr__(self):
        if not self.patterns or not self.on:
            return str(self.meter)
        marks = "".join(str(i) for i in sorted(self.on))
        inner = "+".join(repr(p) for p in sorted(self.patterns,
                                                 key=lambda p: p.meter))
        return f"({self.meter}[{marks}]/{inner})"


def render_node(node, start, end, base_freq, amp_scale, amp, track, sr,
                delay_upbeats=False):
    """Recursively render one pattern into `track` spanning `[start, end]`.

    The node's own `meter` clicks always sound. Its sub-patterns play over
    every beat whose index is in `node.on` (the pulses being fractalized),
    quieter by `amp_scale`, so full patterns nest on top of patterns.
    `delay_upbeats` withholds the node's own "and"s; the forest callers pass it
    for the accent root so its upbeats are re-stamped at the loudness floor.
    """
    span = end - start
    step = span / node.meter
    if st.session_state.get("waves", False):
        wave_amp = st.session_state.get("wave_amp", DEFAULT_WAVE_AMP)
        _add_wave(track, node.meter, start, end, base_freq, sr=sr, amp=amp * wave_amp, rank=node.rank)
    for i in range(node.meter):
        t = start + i * step
        click = make_click(beat_freq(node.meter, base_freq, i, node.rank), sr=sr)
        add_wrapped(track, click * amp, int(t * sr))
        if node.rank == 1:  # the foundation base's beats wear the bright crown
            crown = _make_crown(base_freq, sr=sr) * CROWN_GAIN
            add_wrapped(track, crown * amp, int(t * sr))
        if st.session_state.get("enable_upbeats", DEFAULT_UPBEATS) and not delay_upbeats:
            and_click = make_and_click(and_freq(node.meter, base_freq, node.rank), sr=sr,
                                       meter=node.meter)
            add_wrapped(track, and_click * amp * AND_GAIN, int((t + step / 2) * sr))
        if i in node.on:
            for child in node.patterns:
                render_node(child, t, t + step, base_freq, amp_scale,
                            amp * amp_scale, track, sr)


def render_forest(roots, cycle_len, base_freq, amp_scale=0.6, sr=SR,
                  accent=True, count_words=None):
    """Render a list of root patterns (the base pyramid) into one cycle.

    Each root is a `Node`. All meters at all depths sound; sub-patterns are
    quieter (`amp_scale` per level), and pitch follows the root selection order
    (first picked = foundation, lowest and slightly louder) down every level.
    When
    `accent` is on, the cycle-start indicator (where all beats intersect at
    t=0) is added per the selected cycle-accent option. When the spoken-counts
    toggle is on, the accent base (`roots[0]`) says its numbers over its own
    beat grid; `count_words` is a word→samples dict used by tests, letting the
    app default to the cached gTTS voice.
    """
    one_cycle = np.zeros(int(cycle_len * sr))
    for idx, node in enumerate(roots):
        render_node(node, 0.0, cycle_len, base_freq, amp_scale, 1.0,
                    one_cycle, sr, delay_upbeats=(idx == 0))
    if accent:
        add_cycle_accent(one_cycle, sr=sr)
    if roots and st.session_state.get("enable_counts", DEFAULT_COUNTS):
        add_counts(one_cycle, roots[0].meter, cycle_len, base_freq, sr=sr,
                  words=count_words)
    if roots:  # the accent root's upbeats get their loudness floor before normalize
        add_lead_and_guard(one_cycle, roots[0].meter, cycle_len, base_freq, sr=sr)
    peak = np.max(np.abs(one_cycle))
    if peak > 0:
        one_cycle = one_cycle / peak * 0.9
    return one_cycle


def to_wav_bytes(samples, sr=SR):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        if samples.ndim == 2:  # stereo, shape (2, N) -> true L/R interleave
            wf.setnchannels(2)
            interleaved = samples.T.reshape(-1)
        else:
            wf.setnchannels(1)
            interleaved = samples
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((interleaved * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def play_looped(samples, num_cycles, sr=SR):
    """Render a seamlessly-loopable WAV and play it indefinitely.

    The single-cycle `samples` buffer is tiled `num_cycles` times so the pulse
    is carried across cycle boundaries *inside* the buffer. A browser's
    `<audio>` `loop` can have a small seam at the very end of its buffer; by
    keeping several cycles in one buffer, that seam no longer falls immediately
    after the last beat of every cycle, so the pulse is not interrupted on
    repeat.
    `num_cycles` lets users loop for as long as they like while keeping every
    internal cycle boundary seamless.
    """
    st.audio(to_wav_bytes(np.tile(samples, num_cycles)), format="audio/wav",
             autoplay=True, loop=True)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def meter_color(m):
    """A visually distinct colour per meter, using a categorical palette."""
    tab = plt.get_cmap("tab10")
    return tab((int(m) - 1) % 10)


def draw_timeline(meters, cycle_len):
    fig, ax = plt.subplots(figsize=(8, 0.9 * max(len(meters), 1) + 0.5))
    for row, n in enumerate(sorted(meters)):
        y = len(meters) - row
        positions = np.linspace(0, cycle_len, n, endpoint=False)
        color = meter_color(n)
        ax.hlines(y, 0, cycle_len, color=color, linewidth=2, alpha=0.5)
        ax.scatter(positions, [y] * n, color=color, s=90, zorder=3)
        ax.text(-0.04 * cycle_len, y, f"{n}", va="center", ha="right",
                fontsize=11, fontweight="bold")
    ax.set_xlim(-0.08 * cycle_len, cycle_len)
    ax.set_ylim(0.3, len(meters) + 0.7)
    ax.axis("off")
    ax.set_title("One shared cycle, divided differently by each meter", fontsize=11)
    fig.tight_layout()
    return fig


def draw_forest(roots, cycle_len):
    """Draw the nested patterns as an expanded tree of lines.

    Every pulse (and every sub-pulse) gets its own horizontal line over the
    shared cycle. A base meter is a full-cycle line; each fractalized beat then
    opens the nested polyrhythm as shorter lines directly beneath it, spanning
    that beat. Nothing overlaps and nothing needs blending, so the recursion
    reads as a clear, expandable ladder.
    """
    lines = []  # list of (start, end, meter)

    def expand(node, start, end):
        step = (end - start) / node.meter
        lines.append((start, end, node.meter))
        for i in range(node.meter):
            if i in node.on:
                t = start + i * step
                for child in node.patterns:
                    expand(child, t, t + step)

    for node in roots:
        expand(node, 0.0, cycle_len)

    fig, ax = plt.subplots(figsize=(8, 0.62 * len(lines) + 0.6))
    ymax = len(lines) - 1
    for r, (start, end, meter) in enumerate(lines):
        y = ymax - r
        color = meter_color(meter)
        beats = np.linspace(start, end, meter, endpoint=False)
        ax.hlines(y, start, end, color=color, linewidth=3, alpha=0.55)
        ax.scatter(beats, [y] * meter, color=color, s=70, zorder=3)
        ax.text(start, y, f" {meter}", va="center", ha="left", fontsize=9,
                color=color, fontweight="bold")

    ax.set_xlim(-0.02 * cycle_len, cycle_len)
    ax.set_ylim(-0.6, ymax + 1.0)
    ax.axis("off")
    ax.set_title("Fractal: every pulse on its own line, "
                 "sub-pulses expand beneath the beat that carries them",
                 fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Toro's Harmonic Rhythm", layout="centered")
st.title("Efrain Toro's harmonic perspective of rhythm")

st.markdown(
    """
Percussionist and theorist **Efrain Toro** teaches that instead of counting
one meter and deriving another from it (the *linear* approach), you can hear
several meters as **independent layers occupying the same span of time** —
like overtones in a harmonic series. Here, each meter's clicks are literally
pitched at that harmonic of a shared fundamental, so you hear the metaphor,
not just the beat.
"""
)

st.sidebar.header("Settings")
cycle_len = st.sidebar.slider("Cycle length ('one'), seconds", 1.0, 8.0, 2.0, 0.25,
                             key="cycle_len")
base_freq = st.sidebar.slider("Fundamental (lowest) frequency (Hz)",
                              20, 140, 40, 5)
sound = st.sidebar.radio(
    "Sound",
    options=list(_SOUNDS),
    index=0,
    key="sound",
)
st.sidebar.toggle(
    "Waves under the clicks",
    value=st.session_state.get("waves", False),
    key="waves",
    help="Hear each meter's wave audibly: a pitched tone that breathes beneath "
         "the clicks — full loudness at each beat, sinking to silence in the "
         "trough — so the rise and fall of every wave is heard, not just the "
         "crest where the click sits.",
)
if st.session_state.get("waves", False):
    st.sidebar.slider(
        "Wave volume",
        min_value=0.0, max_value=1.0, value=DEFAULT_WAVE_AMP, step=0.05,
        key="wave_amp",
        help="How loud the underlying wave is relative to the clicks.",
    )
st.sidebar.toggle(
    "Play 'and' between each beat",
    value=st.session_state.get("enable_upbeats", DEFAULT_UPBEATS),
    key="enable_upbeats",
    help="Add a light, high 'and' — the 'and' of 'one and, two and' — "
         "halfway between a pulse's beats: a soft pitched brush clearly above "
         "its beat, ringing at the upbeat pitch. The accent meter's 'and's "
         "hold a loudness floor, so the lead stays audible in a full stack.",
)
if st.session_state.get("enable_upbeats", DEFAULT_UPBEATS):
    st.sidebar.slider(
        "Upbeat pitch factor",
        min_value=1.05, max_value=3.0, value=2.0, step=0.05,
        key="upbeat_factor",
        help="How high the 'and' between beats rings relative to the beats "
             "(1.05 = barely higher, 3.0 = very high).",
    )
st.sidebar.toggle(
    "Spoken counts (the accent says the numbers)",
    value=st.session_state.get("enable_counts", DEFAULT_COUNTS),
    key="enable_counts",
    help="A voice counts over the rhythm: the accent meter says 'one, and, two, "
         "and…' over its own beats — the lead meter's count riding the groove.",
)
st.sidebar.toggle(
    "Alternate the accent every N cycles",
    value=st.session_state.get("rotate_accent", False),
    key="rotate_accent",
    help="Hand the accent (and the spoken count, when on) from base to base "
         "within one clip: each base leads for a turn, then the next — the "
         "radio's pick starts the rotation, the rest follow your selection order.",
)
if st.session_state.get("rotate_accent", False):
    st.sidebar.slider(
        "Cycles per accent (turn length)",
        min_value=1, max_value=4, value=2, step=1,
        key="rotate_turn_len",
        help="How many cycles each base leads before the accent hands off. The "
             "clip always gives every base its turn, so its minimum length is "
             "bases × this many cycles.",
    )
cycle_accent = st.sidebar.radio(
    "Cycle start indicator",
    options=["Off"] + list(_CYCLE_ACCENTS),
    index=0,
    key="cycle_accent",
    help="Add a distinct sound at the start of every cycle — where all the "
         "meters' first beats intersect — to mark the 'one'. The individual "
         "meter sounds continue to play on top.",
)
st.sidebar.toggle(
    "Stereo (split meters L/R)",
    value=st.session_state.get("stereo", False),
    key="stereo",
    help="Spread the base pyramid across two channels: half the meters on the "
         "left, half on the right. The cycle-start indicator stays centred on "
         "both. Affects the base pyramid (sections 1 & 2).",
)
num_cycles = st.sidebar.slider(
    "Cycle repeats per buffer (loop duration)",
    min_value=1, max_value=512, value=4, step=1,
)
st.sidebar.caption(
    f"Plays continuously for ~{num_cycles * cycle_len:.0f}s per buffer; "
    "each cycle loops seamlessly into the next."
)


st.divider()

# --- Section 1: two against three -----------------------------------------
st.header("1. The foundation: two against three")
st.markdown(
    "Toro treats **2** and **3** as the bedrock relationship — the beat "
    "splits only in half, the simplest possible division. Try each alone, "
    "then together as two coexisting lines rather than one derived from "
    "the other."
)

col1, col2, col3 = st.columns(3)
if col1.button("Play 'two' alone"):
    play_looped(render_combined([2], cycle_len, base_freq), num_cycles)
    st.pyplot(draw_timeline([2], cycle_len))
if col2.button("Play 'three' alone"):
    play_looped(render_combined([3], cycle_len, base_freq), num_cycles)
    st.pyplot(draw_timeline([3], cycle_len))
if col3.button("Play 'two' + 'three' together"):
    play_looped(render_combined([2, 3], cycle_len, base_freq), num_cycles)
    st.pyplot(draw_timeline([2, 3], cycle_len))

st.divider()

# --- Section 2: build the harmonic pyramid ---------------------------------
st.header("2. Build the harmonic pyramid")
st.markdown(
    "In *The Odd in You*, Toro has the feet hold **2** and **3** while the "
    "hands work through meters **1–9**, each one a new layer added to the "
    "same cycle. Select any combination below and hear them stacked."
)

default_meters = [2, 3]
chosen = st.multiselect(
    "Active meters (pick the accent below to choose who leads)",
    options=list(range(1, 10)),
    default=default_meters,
)

choose_col, accent_col = st.columns(2)

with accent_col:
    if chosen:
        accent = st.radio(
            "Accent meter (the foundation — lowest, louder, crowned)",
            options=list(chosen),
            key="pyramid_accent",
        )

with choose_col:
    if st.button("Play selected pyramid", type="primary"):
        if not chosen:
            st.warning("Select at least one meter first.")
        elif st.session_state.get("rotate_accent", False):
            turn_len = st.session_state.get("rotate_turn_len", 2)
            order = lead_with(chosen, accent)
            clip = render_rotated(chosen, cycle_len, base_freq,
                                  turn_len=turn_len, accent=accent)
            play_looped(clip, num_cycles)
            st.pyplot(draw_timeline(chosen, cycle_len))
            st.caption(
                f"Meters {', '.join(map(str, chosen))} all fill the same "
                f"{cycle_len:.2f}s cycle; the accent rotates through "
                f"{', '.join(map(str, order))} every {turn_len} cycle(s) — "
                f"**{accent}** leads first — each base lowest and slightly "
                "louder during its turn, the rest following your selection "
                "order above it."
            )
        else:
            ordered = lead_with(chosen, accent)
            play_looped(render_combined(ordered, cycle_len, base_freq), num_cycles)
            st.pyplot(draw_timeline(ordered, cycle_len))
            st.caption(
                f"Meters {', '.join(map(str, chosen))} all fill the same "
                f"{cycle_len:.2f}s cycle, each subdividing it into that many "
                "equal parts; the accent — "
                f"**{accent}** — is the foundation, lowest and slightly "
                "louder, and the rest follow in your selection order above it."
            )

st.divider()
st.markdown(
    "**Tip:** start with just `2` and `3`, get comfortable, then add `4` "
    "and `6` (their doubles), then try the harder `5` and `7` — the "
    "progression Toro recommends for building this perception gradually."
)

st.divider()

# --- Section 3: rhythm fractals --------------------------------------------
st.header("3. Rhythm fractals")
st.markdown(
    "A fractal layers **full patterns on top of other patterns**. Pick the "
    "base polyrhythm (the 2 and the 3, say), then choose **whose beats** carry "
    "the recursion — the 2-beats, the 3-beats, or both. A **nesting depth** "
    "slider sets how many layers deep it recurs: depth 1 plays the base "
    "polyrhythm once inside each chosen beat, depth 2 repeats it inside those, "
    "and so on. All meters always sound; layers are quieter, and a radio picks "
    "the **accent base** — the foundation of every layer, lowest and slightly "
    "louder — with the rest following in your selection order at every depth."
)

amp_scale = st.slider("Layer amplitude scaling (per nested level)",
                      0.2, 1.0, 0.6, 0.05)

MAX_DEPTH = 4


def build_fractal(bases, depth, on_by_meter):
    """Build the nested pattern forest from the base meters, a depth, and the
    per-base-meter set of pulses to fractalize.

    The roots are the base meters themselves (the base polyrhythm, e.g. 2 and
    3). At the chosen `depth`, the *entire base polyrhythm* recurs inside the
    selected pulses of each base meter, one level at a time, until `depth`
    layers are nested. The same relative pulse choices (`on`) are carried down
    each level, so deeper recursion fractalizes the same beats proportionally.
    """
    bases = bases if bases else []

    def build(meter, d, on):
        node = Node(meter, on=on, rank=order[meter])
        if d > 0 and bases:
            node.patterns = [build(b, d - 1, on) for b in bases]
        return node

    order = {b: i + 1 for i, b in enumerate(bases)}

    return [build(b, depth, on_by_meter.get(b, set())) for b in bases]


def count_nodes(roots):
    total = 0
    for n in roots:
        total += 1
        total += count_nodes(n.patterns)
    return total


# The base polyrhythm: which meters form the bottom layer (2 and 3, say).
chosen_bases = st.multiselect(
    "Base meters (the base polyrhythm)",
    options=list(range(2, 10)),
    default=[2, 3],
)

depth = st.slider(
    "Nesting depth (how many layers deep the base polyrhythm recurs)",
    min_value=0, max_value=MAX_DEPTH, value=1, step=1,
)

# Which base meter's beats carry the fractal. E.g. with a 2-against-3 base,
# pick the 2-beats, the 3-beats, or both.
on_by_meter = {}
if chosen_bases:
    on_meters = st.multiselect(
        "Play the base polyrhythm on the beats of which meter(s)?",
        options=sorted(chosen_bases),
        default=sorted(chosen_bases),
        key="fractal_on_meters",
        help=f"Choose which base meter's beats carry the nested rhythm. For a "
             f"{' against '.join(map(str, sorted(chosen_bases)))} base, "
             "selecting only the 2 plays 2v3 on each 2-beat; only the 3 plays "
             "it on each 3-beat; both plays it on every beat of both.",
    )
    for m in on_meters:
        on_by_meter[m] = set(range(m))

roots = build_fractal(chosen_bases, depth, on_by_meter)
n_nodes = count_nodes(roots)

st.markdown(
    "**The fractal:** "
    + " &nbsp;·&nbsp; ".join(repr(r) for r in roots)
    + (f" &nbsp;·&nbsp; _depth {depth}_" if chosen_bases else "")
)

if not chosen_bases:
    st.warning("Select at least one base meter first.")

fractal_col, fractal_accent_col = st.columns(2)

with fractal_accent_col:
    if chosen_bases:
        accent_base = st.radio(
            "Accent base (the foundation of every layer)",
            options=list(chosen_bases),
            key="fractal_accent",
        )

with fractal_col:
    if st.button("Play fractal loop", type="primary"):
        if not chosen_bases:
            st.warning("Select at least one base meter first.")
        elif st.session_state.get("rotate_accent", False):
            turn_len = st.session_state.get("rotate_turn_len", 2)
            order = lead_with(chosen_bases, accent_base)
            turn_roots = []
            oversize = False
            for m in order:
                roots = build_fractal(lead_with(chosen_bases, m), depth, on_by_meter)
                if count_nodes(roots) > 4000:
                    oversize = True
                    break
                turn_roots.append(roots)
            if oversize:
                st.warning(
                    f"Depth {depth} with bases {chosen_bases} builds an accent "
                    "rotation with too many pulses to render clearly. Reduce "
                    "depth, base meters, or turn-length."
                )
            else:
                parts = [render_forest(r, cycle_len, base_freq,
                                       amp_scale=amp_scale) for r in turn_roots]
                play_looped(_finish_rotation(parts, turn_len), num_cycles)
                st.pyplot(draw_forest(turn_roots[0], cycle_len))
                st.caption(
                    "Base polyrhythm: "
                    + ", ".join(map(str, chosen_bases))
                    + f" over the {cycle_len:.2f}s cycle, nested {depth} level(s) "
                    "deep on the selected pulses; the accent base rotates through "
                    f"{', '.join(map(str, order))} every {turn_len} cycle(s) — "
                    f"**{accent_base}** leads first — each base lowest and "
                    "slightly louder during its turn, at every depth."
                )
        else:
            ordered_bases = lead_with(chosen_bases, accent_base)
            ordered_roots = build_fractal(ordered_bases, depth, on_by_meter)
            if count_nodes(ordered_roots) > 4000:
                st.warning(
                    f"Depth {depth} with meters {ordered_bases} builds "
                    f"{count_nodes(ordered_roots)} pulses — too many to render "
                    "clearly. Reduce depth or base meters."
                )
            else:
                play_looped(render_forest(ordered_roots, cycle_len, base_freq,
                                          amp_scale=amp_scale), num_cycles)
                st.pyplot(draw_forest(ordered_roots, cycle_len))
                st.caption(
                    "Base polyrhythm: "
                    + ", ".join(map(str, chosen_bases))
                    + f" over the {cycle_len:.2f}s cycle, nested {depth} level(s) "
                    "deep on the selected pulses; the accent — "
                    f"**{accent_base}** — is the foundation of every layer, "
                    "lowest and slightly louder, and the rest follow in your "
                    "selection order above it."
                )

