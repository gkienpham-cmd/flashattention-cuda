# Master prompt for Claude Design

Paste everything under the line into Claude Design (claude.ai). It references the files in this
`flash-website/` folder — attach `content/`, `design/design-tokens.md`, and `assets/` so the tool has
the real data. Generate the components, then drop them back into `flash-website/` and sync with
`/design-sync`.

---

You are my design engineer. Build an **award-level personal portfolio website** for me, Kien Pham — a
mechanical-engineering student and ML-systems builder. Target quality: Awwwards / design-agency
level, not a template. The goal is to land ML-systems / GPU-kernel / research internships, and to
**wow in the first 10 seconds**. Everything below is the spec; the real content is in the attached
`content/`, `design/design-tokens.md`, and `assets/` files — use them verbatim, don't invent numbers.

## Who I am / the lead framing
Lead with the **mechanical-engineer → ML crossover**. The through-line, learned racing go-karts and
now applied to silicon: **"find the limiting factor, and trim time against a measurement."** I rebuilt
**FlashAttention from scratch — 17 CUDA kernels across 4 GPU architectures (T4 → B300)** — using a
roofline model to predict the bottleneck *before* writing each kernel, then measuring, then recording
where the prediction was wrong. The honest misses are a feature, not a bug: they're the strongest
evidence the wins are real. Full bio, voice, and pull-quotes are in `content/narrative.md`.

## Tech stack (use exactly this)
- **Next.js (App Router) + TypeScript**
- **Tailwind CSS** with the tokens in `design/design-tokens.md`
- **Framer Motion** for subtle reveals/micro-interactions
- **GSAP + ScrollTrigger** for the big pinned scroll scenes
- **Lenis** for smooth scroll
- Fonts via **`next/font`** (Space Grotesk display, Inter body, JetBrains Mono for numbers/code)
- Output **individual, self-contained `.tsx` components** + a page that composes them, so I can drop
  them into this `flash-website/` folder and sync. No backend.

## Visual identity (from `design/design-tokens.md` — follow it)
Dark, warm **stone** palette (bg `#0C0A09`, surfaces `#1C1917`/`#292524`, text `#F5F5F4`), muted so
the work shines. A small set of complementary, low-saturation accents mapped to the three eras of the
journey: **teal `#5EAAA8` (prefill), amber `#E0A458` (decode), clay `#C97B63` (frontier)**, plus sage
/ violet for extra data series. Accents used sparingly (a line, a dot, one word) — never big fills.

## Non-negotiable global rules (award-site craft)
1. **Every section is 100vh** (`min-height: 100dvh`) with generous padding — give it room to breathe.
2. **Media renders at full opacity — never a dark overlay.** For legibility of text over media, use a
   **gradient mask** (solid → transparent), so copy emerges from the scene rather than sitting on a scrim.
3. **Text can sit *behind* the hero subject** (layered) for depth.
4. **Unique embedded fonts** (above), not system defaults.
5. **Buttery-smooth, no-lag, fully mobile-responsive.** The hero uses a **canvas frame-scrubber**
   (preload an image sequence, draw the frame matching scroll progress) — NOT a heavy `<video>` — so
   it never stutters. Debounce/throttle scroll work; preload frames; use `requestAnimationFrame`.
6. **Honor `prefers-reduced-motion`** — swap every scrub/count-up for a clean static state.
7. Motion is **purposeful**: it should explain (draw the roofline, plot the kernels), not just decorate.

## The signature hero — cinematic KART → GPU morph (`ScrollSequenceHero`)
This is the 10-second wow. A **GSAP-pinned `<canvas>`** scrubs an image sequence as I scroll:
**frame 1 = a go-kart on a racing line → the track lines morph into circuit traces → a GPU die → the
B300 board.** It renders my whole thesis literally: *the same bottleneck-hunting discipline, now on
silicon.*
- Preload frames from `assets/hero-sequence/frames/frame_0001.jpg …` (see that folder's README for how
  I generate them). If the sequence isn't present yet, **fall back gracefully** to an animated-roofline
  hero (below) using `assets/hero-sequence/poster.jpg`.
- **Overlay, drawn on top of the scrub:** the **roofline** draws itself — axes fade in, the ridge line
  sweeps across, and the **v1→v12 kernel dots plot onto the curve** as the headline numbers count up
  (`content/metrics.json → headline`: 17 kernels · 4 architectures · 8–16× · 1785 TFLOP/s).
- **Kinetic title** "FlashAttention, from scratch" sits **behind the subject** through a gradient mask;
  a mono line types in: *"Find the limiting factor. Trim time against a measurement."*
- Subtle **mouse-parallax** on the layers. Scroll releases the pin and hands off into the About section.

## Site map — 9 sections, each 100vh (copy in `content/site-content.md`)
1. **Hero** — `ScrollSequenceHero` (above).
2. **About** — "From the racing line to the roofline." The MechE→ML crossover + Sconce
   (model-compression research) + the TGC Speedway app. A small stat row.
3. **The Journey (v1→v12)** — a **scroll-driven timeline** from `content/kernels.json`. Three
   era-colored bands (teal/amber/clay); each kernel is a station that reveals its technique, finding,
   and headline as it enters. Include the **"4 dead ends → the win"** beat in Era 2 (tensor cores,
   double-buffer, occupancy, ILP all NULL — rendered in `danger`, then v8.7 lands the win in an accent).
4. **Benchmark terminal** — `BenchmarkTerminal`: a pinned section that **types out a benchmark run**
   line by line while the headline numbers **count up** as they print (script in `site-content.md §4`).
5. **Results** — animated **speedup progression** + **memory/capacity wins** (125× / 3.56× / 202×) +
   the roofline scorecard. Use `assets/diagrams/v1-v12-arc-summary.svg`, `v12-throughput-regime.svg`.
6. **Independent Validation** — the reproduced-AIs scorecard (234.8 / 469.7 / 835.0 / 3.6, all exact)
   + confirmed-claims checklist + the "honesty is the strongest evidence" pull-quote. Use
   `assets/diagrams/portfolio-validation-scorecard.svg`. Keep the proxy-grade / single-seed honesty note.
7. **Methodology / Honest Misses** — "The misses are the method." Big **kinetic pull-quotes** from
   `content/narrative.md` (the L2 lesson, the S-elimination paradox, the v12 correction).
8. **Diagram reel** — `DiagramReel`: a **horizontal scroll/scrub montage** of the SVGs in
   `assets/diagrams/`.
9. **Contact** — "Let's build fast things." GitHub (github.com/gkienpham-cmd/flashattention-cuda),
   Email (pgkien11@gmail.com), LinkedIn (`{{LINKEDIN_URL}}` placeholder). Primary resume bullet =
   `content/resume.md` Version C.

## Components to generate (name them exactly)
- `SmoothScrollProvider` — Lenis wrapper.
- `ScrollSequenceHero` — canvas frame-scrubber + roofline overlay + masked kinetic title (the signature).
- `RooflineOverlay` — the self-drawing roofline (axes → ridge → v1..v12 dots), reusable in the hero + Results.
- `JourneyTimeline` + `KernelStation` — scroll-reveal timeline from `kernels.json`, era-colored.
- `DeadEndsToWin` — the four-NULL-then-win beat.
- `BenchmarkTerminal` — typewriter stream + count-up.
- `CountUp` — in-view number animation (mono).
- `SpeedupChart` — animated progression.
- `ValidationScorecard` — reproduced-AIs table reveal.
- `KineticHeadline` — split-text GSAP headline.
- `DiagramReel` — horizontal scrub of SVGs.
- `SectionReveal` (100vh wrapper + gradient-mask helper) · `MagneticButton` · `CursorGlow`.

## Data & content contract
- Numbers: `content/metrics.json`. Journey: `content/kernels.json`. Copy: `content/site-content.md`.
  Voice + quotes: `content/narrative.md`. Bullets: `content/resume.md`. Tokens: `design/design-tokens.md`.
- **Do not invent or round differently** — every figure is measured and traceable. Preserve the honesty
  notes (v5 bench unmeasured; 4× cuBLAS gap is inferred; Blackwell verdicts proxy-grade; RMSE single-seed).

Deliver the components and the composed page. Prioritize the `ScrollSequenceHero` + `JourneyTimeline`
first — those are the ones that have to feel award-level.
