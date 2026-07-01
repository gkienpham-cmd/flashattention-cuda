# flash-website

The build kit for my portfolio site — an award-level, scroll-driven showcase of the
FlashAttention-from-scratch project. This folder holds everything Claude Design needs to generate the
animated components, plus the components once they land.

## What's here

```
claude-design-prompt.md      ★ paste this into Claude Design to generate the components
content/
  site-content.md            all section copy (9 sections)
  kernels.json               v1→v12 journey data (drives the timeline)
  metrics.json               headline numbers (drive the count-ups)
  narrative.md               the MechE→ML crossover story + pull-quotes
  resume.md                  the 3 resume bullets (Version C leads)
design/
  design-tokens.md           stone palette + era-accents, type, spacing, motion
assets/
  diagrams/                  12 hand-authored SVGs used across the site
  hero-sequence/             the cinematic KART→GPU morph (see its README) + poster.jpg fallback
    frames/                  drop frame_0001.jpg … here
```

## How to use it

1. **Generate the components.** Open [claude.ai/design](https://claude.ai/design), paste
   `claude-design-prompt.md`, and attach `content/`, `design/design-tokens.md`, and `assets/`. It will
   produce Next.js + TypeScript components (Framer Motion + GSAP + Lenis) styled to the tokens.
2. **Generate the hero morph** (optional, do it anytime) — follow `assets/hero-sequence/README.md` to
   make the kart → GPU-die scroll sequence, and drop the frames in `assets/hero-sequence/frames/`.
   Until then the hero falls back to the animated-roofline hero over `poster.jpg`.
3. **Drop the generated components into this folder** and wire up a Next.js app around them.
4. **Sync** with `/design-sync` to keep this local library and the Claude Design project in step.

## Design intent (one paragraph)

Dark, warm **stone** palette — muted so the work shines — with three complementary era-accents
(teal / amber / clay) that carry meaning. Lead with the **mechanical-engineer → ML crossover**:
*"find the limiting factor, trim time against a measurement."* The 10-second wow is a scroll-scrubbed
**kart → GPU-die morph** with the roofline drawing itself on top. Every section 100vh; full-opacity
media with gradient masks (never dark overlays); smooth Lenis scroll; fully mobile. Every number is
measured and traceable to the repo's `docs/results.md` — the honest misses stay on display.

## Provenance

Content is extracted from the parent repo (`docs/portfolio.md`, `docs/results.md`,
`docs/interview-prep.md`) and the diagrams from `docs/diagrams/`. Don't invent or re-round numbers.
