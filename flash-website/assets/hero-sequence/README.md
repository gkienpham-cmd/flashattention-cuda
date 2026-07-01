# Hero sequence — the KART → GPU morph (asset-generation guide)

The hero is a **scroll-scrubbed image sequence** (the award-site technique). You generate a short
morph video, export it to a numbered JPEG sequence, drop the frames in `frames/`, and the
`ScrollSequenceHero` canvas component scrubs through them tied to scroll. This is how the no-lag award
sites do it — a `<canvas>` drawing the frame that matches scroll progress, not a `<video>`.

## The concept
**Frame 1:** a go-kart on a racing line (top-down or dramatic 3/4), warm track, clean composition —
your bottleneck-hunting origin.
**Frame 2:** the track lines have morphed into circuit traces → a **GPU die** → the **B300 board**.
Same palette (warm stone, muted), so the two frames share a mood and the morph feels continuous.

The message: *the same discipline — find the limiting factor — on silicon now.*

## Steps (any tools; these are examples, not endorsements)
1. **Generate frame 1** with an image model (e.g. GPT-Image / Midjourney / Ideogram): a go-kart on a
   racing line, warm muted palette matching `design/design-tokens.md`, cinematic, negative space at
   the top for the title. Aspect 16:9, high-res.
2. **Generate frame 2** from frame 1 as reference: "keep the same scene and palette, but morph the
   track lines into glowing circuit traces resolving into a GPU die / NVIDIA B300 board." Ask for a
   continuation, not a fresh scene, so the two frames line up.
3. **Animate the morph** frame 1 → frame 2 with an image-to-video / interpolation tool (e.g. Kling,
   Runway, Luma, Sora). Prompt: *"animate a smooth cinematic morph from frame one to frame two,
   natural and aesthetic, the racing line dissolving into circuit traces and a GPU die."* ~4–8s, 1080p.
4. **Export to a JPEG sequence.** Either an online "video → JPEG sequence" converter, or locally:
   ```bash
   # ~30 frames/sec of video → numbered JPEGs (needs ffmpeg)
   ffmpeg -i morph.mp4 -vf "fps=30,scale=1600:-1" frames/frame_%04d.jpg
   ```
   Aim for **60–120 frames total** (enough to feel smooth when scrubbed; few enough to preload fast).
   Keep each JPEG ≲ 150 KB (quality ~6 in ffmpeg: add `-q:v 6`).
5. **Drop the frames in `frames/`** named `frame_0001.jpg … frame_00NN.jpg` (zero-padded, sequential).
6. **Add a `poster.jpg`** here (frame 1, full quality) — the static fallback shown before the sequence
   loads and on `prefers-reduced-motion`.

## Contract the component expects
- `frames/frame_%04d.jpg`, 1-indexed, contiguous.
- `poster.jpg` in this folder.
- The component reads the count from a small manifest or just probes until a frame 404s — tell Claude
  Design which; simplest is to pass `frameCount` as a prop and set it to however many you exported.

## Until you generate it
The site works without the sequence: `ScrollSequenceHero` falls back to the **animated-roofline hero**
over `poster.jpg`. So you can build and ship the whole site first, then swap the morph in later.
