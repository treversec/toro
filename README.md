# Toro's Harmonic Rhythm

An interactive polyrhythm app based on percussionist and theorist [Efrain
Toro](https://rhythmline.net/about-efrain/)'s *harmonic
perspective of rhythm* — built with [Streamlit](https://streamlit.io/).

Instead of counting one meter and deriving another from it (the *linear*
approach), different meters are heard as **independent layers occupying the
same span of time** — like overtones in a harmonic series. Here, each
meter's clicks are *literally* pitched at that harmonic of a shared
fundamental, so you hear the metaphor, not just the beat.

🎧 **[Try the live app →](https://aiforet.streamlit.app/)**

## Features

- **Section 1 — The foundation: two against three.** Hear `2` and `3` each
  alone, then together as two coexisting lines rather than one derived from
  the other.
- **Section 2 — The harmonic pyramid.** Stack any combination of meters
  1–9 over the same cycle — the polyrhythm exercise from Toro's *The Odd in
  You*, where the feet hold `2` and `3` while the hands work through meters
  1–9.
- **Section 3 — Rhythm fractals.** Layer full patterns on top of other
  patterns: pick the base polyrhythm, choose which beats carry the
  recursion, and nest it as deep as 4 levels. Every meter always sounds;
  each layer is quieter, and pitch follows your base *selection order* at
  every nesting depth — the first base you pick is the foundation of the
  whole structure — with a paired **Play reversed** button for the instant
  flip.
- **Rich, configurable sound.** Pitched beeps or synthesized drums,
  a separate low-downbeat/cycle-start accent ("deep drum" or "high chime"),
  an optional "and" between each beat, and an optional "waves under the
  clicks" mode where each beat's audible swell rings at that pulse's own
  frequency. The *selection order* is the perspective — in the pyramid and in
  the fractals alike: the first meter you pick is the foundation — lowest and
  slightly louder — and each later pick perches one rung higher; a paired
  **Play reversed** button flips it in one tap for instant before/after
  perspective.
- **Seamless looping.** Every cycle loops cleanly into the next, with
  configurable repetition counts for longer listening buffers.
- **Stereo spread.** Split the base pyramid across left/right channels to
  spatially separate the meters.

## Quick start

```bash
pip install -r requirements.txt
streamlit run toro.py
```

Requires Python 3.9+, `streamlit`, `numpy`, and `matplotlib` (see
[`requirements.txt`](requirements.txt)).

The app also deploys as-is to [Streamlit Community Cloud](https://streamlit.io/cloud).

## How it works

`toro.py` is a single self-contained Streamlit app. The audio is synthesized
on the fly: for each meter `n`, the cycle is subdivided into `n` evenly
spaced clicks, and each click is pitched at the `n`th harmonic of a
fundamental frequency you can set. The rhythm fractals are built from a
recursive pattern tree (`Node`), where each sub-pattern plays — quieter and
pitched by the base selection order at every depth — over every beat it's
attached to.

Click positions are rendered with wrap-around at the loop point, so a click
that would overhang the end of a cycle is folded back onto the start. The
resulting single cycle tiles seamlessly, keeping the underlying pulse
unbroken even across `<audio>` loop boundaries.

## Project structure

```
toro.py            The entire app: audio synthesis, visualizations, and UI
requirements.txt   Python dependencies
.streamlit/        Streamlit theme and server settings
```

## Credits

The musical idea belongs to [Efrain Toro](https://rhythmline.net/about-efrain/);
this app is an independent demonstration of his teaching. See *The Odd in
You* for the 2-and-3-backed pyramid exercise he describes.

## License

[MIT](LICENSE) © 2026 treversec
