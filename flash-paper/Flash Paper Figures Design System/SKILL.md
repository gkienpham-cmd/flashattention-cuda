---
name: flash-paper-design
description: Use this skill to generate well-styled academic paper figures and assets for the flash-paper PMBS@SC26 submission (GPU attention-decode roofline characterization), either for the paper itself or for talks/posters/mocks. Contains the house figure style — colors, type scale, panel/chip/verdict conventions — plus the 8 canonical figures and reusable primitives.
user-invocable: true
---

Read the README.md file within this skill, and explore the other available files.
If creating visual artifacts (figures, slides, poster panels, diagrams), copy assets out and create static HTML or standalone SVG files for the user to view — SVG text must stay as real <text> elements. If working on the paper repo, export figures over `figures/<name>.svg` keeping basenames unchanged.
Hard rules: white background, flat print-safe palette, 13px text floor, copy is verbatim (never invent or round numbers — Figs 4–5 encode measured data as bar geometry and must stay exactly proportional), no emoji, no gradients or shadows.
If the user invokes this skill without any other guidance, ask them what they want to build or design, ask some questions, and act as an expert designer who outputs HTML artifacts _or_ production code, depending on the need.
