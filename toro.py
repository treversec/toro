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


_SOUNDS = {"Drums (synthesized)": _click_drum,
           "Pitched beeps (original)": _click_beep}


def make_click(freq, sr=SR, dur=0.09, decay=22):
    """Generate one note using the currently selected sound.

    `st.session_state["sound"]` holds the chosen sound name (see `_SOUNDS`);
    falls back to the drums if unset. The chosen generator keeps the `freq`
    argument, so meter n stays pitched at its mapped frequency either way.
    """
    name = st.session_state.get("sound", "Drums (synthesized)")
    return _SOUNDS.get(name, _click_drum)(freq, sr=sr, dur=dur, decay=decay)


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


def meter_freq(n, base_freq):
    """Map a meter to its drum frequency, ascending gently with compression.

    Strict harmonics (`base_freq * n`) leap a full overtone per meter and climb
    too fast for low drums. Instead, low meters stay close together and the
    climb softens as we ascend, so the base of the pyramid sits low while
    keeping a rising, still-distinct pitch per meter.
    """
    return base_freq * (1 + 0.65 * (n - 1) ** 0.8)


def render_meter_track(n, cycle_len, base_freq, sr=SR):
    """n evenly spaced clicks across one cycle, pitched at the nth harmonic.

    Returns a single cycle built to loop seamlessly: a click that would
    overhang the end of the cycle is wrapped back onto the start, so the
    waveform is continuous across the loop point and the underlying pulse
    is never broken by a seam or pause.
    """
    total = int(cycle_len * sr)
    track = np.zeros(total)
    click = make_click(meter_freq(n, base_freq), sr=sr)
    positions = np.linspace(0, cycle_len, n, endpoint=False)
    for p in positions:
        add_wrapped(track, click, int(p * sr))
    return track


def render_combined(meters, cycle_len, base_freq, sr=SR):
    one_cycle = np.zeros(int(cycle_len * sr))
    for n in meters:
        one_cycle += render_meter_track(n, cycle_len, base_freq, sr=sr)
    peak = np.max(np.abs(one_cycle))
    if peak > 0:
        one_cycle = one_cycle / peak * 0.9
    return one_cycle


# ---------------------------------------------------------------------------
# Rhythm fractals (patterns layered on patterns)
# ---------------------------------------------------------------------------
# A fractal is a rhythm-tree: every node is a meter that sounds on its own,
# and it may carry a *sub-pattern* that plays over every one of its beats.
# So "3/2 on top of the 2" is a 2-base with a 3/2 pattern (a 3-base carrying a
# 2) layered over each of its beats. Every level's own meter always sounds;
# sub-patterns are quieter and pitched at their own harmonic, so the layers
# read like overtones nested at a smaller scale.

class Node:
    """One pattern: `meter` clicks, plus a *combination* of sub-patterns played
    over some of those clicks (arbitrarily deep). `on` is the set of beat
    indices (0-based) whose pulse carries the sub-patterns; empty `on` or empty
    `patterns` means "no sub-layer on any beat"."""
    def __init__(self, meter, patterns=None, on=None):
        self.meter = meter
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


def render_node(node, start, end, base_freq, amp_scale, amp, track, sr):
    """Recursively render one pattern into `track` spanning `[start, end]`.

    The node's own `meter` clicks always sound. Its sub-patterns play over
    every beat whose index is in `node.on` (the pulses being fractalized),
    quieter by `amp_scale`, so full patterns nest on top of patterns.
    """
    span = end - start
    step = span / node.meter
    click = make_click(meter_freq(node.meter, base_freq), sr=sr)
    for i in range(node.meter):
        t = start + i * step
        add_wrapped(track, click * amp, int(t * sr))
        if i in node.on:
            for child in node.patterns:
                render_node(child, t, t + step, base_freq, amp_scale,
                            amp * amp_scale, track, sr)


def collect_meters(node, acc):
    acc.add(node.meter)
    for c in node.patterns:
        collect_meters(c, acc)
    return acc


def render_forest(roots, cycle_len, base_freq, amp_scale=0.6, sr=SR,
                  accent=True):
    """Render a list of root patterns (the base pyramid) into one cycle.

    Each root is a `Node`. All meters at all depths sound; sub-patterns are
    quieter (`amp_scale` per level) and pitched at their own harmonic. When
    `accent` is on, the "one" of the cycle (the first beat at t=0) is given an
    extra, pitched impulse at the lowest harmonic present in the pyramid, so
    the downbeat stands out.
    """
    one_cycle = np.zeros(int(cycle_len * sr))
    for node in roots:
        render_node(node, 0.0, cycle_len, base_freq, amp_scale, 1.0,
                    one_cycle, sr)
    if accent and roots:
        meters = set()
        for n in roots:
            collect_meters(n, meters)
        click = make_click(meter_freq(min(meters), base_freq), sr=sr)
        add_wrapped(one_cycle, click * 1.6, 0, sr)
    peak = np.max(np.abs(one_cycle))
    if peak > 0:
        one_cycle = one_cycle / peak * 0.9
    return one_cycle


def to_wav_bytes(samples, sr=SR):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((samples * 32767).astype(np.int16).tobytes())
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
cycle_len = st.sidebar.slider("Cycle length ('one'), seconds", 1.0, 4.0, 2.0, 0.25)
base_freq = st.sidebar.slider("Fundamental (lowest) frequency (Hz)",
                              20, 140, 40, 5)
sound = st.sidebar.radio(
    "Sound",
    options=list(_SOUNDS),
    index=0,
    key="sound",
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
    "Active meters (each played as clicks + its own pitch = nth harmonic)",
    options=list(range(1, 10)),
    default=default_meters,
)

if st.button("Play selected pyramid", type="primary"):
    if not chosen:
        st.warning("Select at least one meter first.")
    else:
        play_looped(render_combined(chosen, cycle_len, base_freq), num_cycles)
        st.pyplot(draw_timeline(chosen, cycle_len))
        st.caption(
            f"Meters {', '.join(map(str, sorted(chosen)))} all fill the same "
            f"{cycle_len:.2f}s cycle, each subdividing it into that many "
            "equal parts and sounding at that harmonic pitch."
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
    "and so on. All meters always sound; layers are quieter and pitched at "
    "their own harmonic."
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
    bases = sorted(bases)

    def build(meter, d, on):
        node = Node(meter, on=on)
        if d > 0 and bases:
            node.patterns = [build(b, d - 1, on) for b in bases]
        return node

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

if st.button("Play fractal loop", type="primary"):
    if not chosen_bases:
        st.warning("Select at least one base meter first.")
    elif n_nodes > 4000:
        st.warning(
            f"Depth {depth} with meters {chosen_bases} builds {n_nodes} "
            "pulses — too many to render clearly. Reduce depth or base meters."
        )
    else:
        play_looped(render_forest(roots, cycle_len, base_freq,
                                  amp_scale=amp_scale), num_cycles)
        st.pyplot(draw_forest(roots, cycle_len))
        st.caption(
            "Base polyrhythm: "
            + ", ".join(map(str, chosen_bases))
            + f" over the {cycle_len:.2f}s cycle, nested {depth} level(s) "
            "deep on the selected pulses; each level is quieter and pitched "
            "at its own harmonic."
        )

