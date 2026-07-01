# Design tokens

Dark, warm **stone** base — muted so the work shines — with a small set of complementary,
low-saturation accents. The three era-accents (teal / amber / clay) map to the three eras of the
journey so color carries meaning, not decoration.

## Color

### Base (warm stone, dark)
| Token | Hex | Use |
|---|---|---|
| `bg` | `#0C0A09` | page background (stone-950) |
| `surface` | `#1C1917` | cards, raised panels (stone-900) |
| `surface-2` | `#292524` | insets, terminal (stone-800) |
| `border` | `#44403C` | hairline rules (stone-700) |
| `text` | `#F5F5F4` | headings (stone-100) |
| `text-body` | `#D6D3D1` | body (stone-300) |
| `text-muted` | `#A8A29E` | captions (stone-400) |
| `text-faint` | `#78716C` | eyebrows, meta (stone-500) |

### Accents (complementary, muted — era-mapped)
| Token | Hex | Meaning |
|---|---|---|
| `teal` (Era 1 · prefill) | `#5EAAA8` | bandwidth wall |
| `amber` (Era 2 · decode) | `#E0A458` | per-CTA wall |
| `clay` (Era 3 · frontier) | `#C97B63` | precision / architecture |
| `sage` (data series) | `#9CA986` | extra chart series / "win" |
| `violet` (data series) | `#9B8CC0` | extra chart series |
| `danger` (dead ends) | `#B45E4D` | the four NULL results |

Rules: accents are used sparingly (lines, dots, a single word), never as large fills. Keep everything
low-saturation so the page reads calm. Media (the hero sequence, diagrams) renders at **full opacity —
no dark overlays.** Text over media uses a **gradient mask**, not a scrim.

## Typography
- **Display / headings:** Space Grotesk (600/700). Tight tracking, large scale.
- **Body:** Inter (400/500).
- **Mono (numbers, code, terminal, data labels):** JetBrains Mono (400/500).
- Embed via `next/font` (Google) so the fonts are self-hosted and unique to the site.

Scale (fluid, `clamp()`): display 3.5–7rem · h2 2–3rem · h3 1.25–1.6rem · body 1–1.15rem ·
mono-data 0.85–0.95rem. Line-height 1.05 for display, 1.6 for body.

## Spacing & layout
- Every section is **100vh** (`min-height: 100dvh`) with generous internal padding (`clamp(3rem,8vh,7rem)`).
- Max content width 72rem; wide data bands full-bleed.
- Hairline rules (`border`) instead of boxes where possible; lots of negative space.

## Motion
- **Smooth scroll:** Lenis (site-wide), `lerp ~0.1`.
- **Reveals:** Framer Motion, `y: 24 → 0`, `opacity 0 → 1`, `ease: [0.16,1,0.3,1]`, in-view once.
- **Pinned scenes:** GSAP ScrollTrigger — the hero canvas scrub, the diagram reel horizontal scrub.
- **Count-up:** trigger on in-view; ease-out; mono font; ~1.2s.
- **Micro:** magnetic buttons, a soft cursor glow, link underlines that wipe in.
- **Respect `prefers-reduced-motion`:** swap scrubs/count-ups for instant states.
- Principle: playful but purposeful. Motion should explain (draw the roofline, plot the kernels),
  not just decorate.

## Accessibility
- Contrast ≥ 4.5:1 for body text on `bg`/`surface` (the stone-300/100 tokens clear this).
- Keyboard-focusable nav + links; visible focus rings in `teal`.
- Reduced-motion path for every scroll-driven scene.
