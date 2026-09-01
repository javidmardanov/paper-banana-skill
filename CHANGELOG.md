# Changelog

## 1.2.0 — 2026-09-01

### Added
- **Multiple providers.** Models are named `provider/model`: `gemini` (default), `openai` (GPT-5.x for reasoning, `gpt-image-2` / `gpt-image-1.5` for rendering, including edit mode), `anthropic` (Claude for the reasoning agents), and `openrouter` (any model, image output via chat modalities). Adapters live in `scripts/providers.py`; only the SDK you use needs installing (`requirements-optional.txt`).
- **Parallel candidates.** `--num-candidates N` renders N images on the first pass, the Critic scores each, and the best seeds the refinement loop (the paper's best-of-N technique).
- **Improve an existing figure.** `--input-image` shows the figure to the Planner (preserve content, improve layout) and passes it to the Visualizer as the edit source.
- **PDF and LaTeX input.** `--methodology-file paper.pdf --pages 3-5` extracts text with pypdf; `.tex` files have comments, citations, refs, and figure environments stripped.
- **Venue style packs.** `--venue` (neurips, iclr, icml, cvpr, acl, aaai) and `--figure-width single|double` feed each venue's column/text width and style notes to the Stylist; `plot_generator.py --venue --width` sizes plots to the same widths. Data in `assets/venues.json`.
- **Resume.** `--resume` reuses Retriever/Planner/Stylist outputs from the work dir and only re-renders.
- **Usage report.** Every run ends with per-model token and image counts.
- **Tests and CI.** 50+ pytest tests (provider adapters against fake SDK clients, mocked end-to-end pipeline, plots) run on every push via GitHub Actions; no API keys needed.

### Changed
- Agents call `common.chat()` / `common.generate_image_bytes()` instead of the Gemini SDK directly.
- `validate_output.py --check-api` checks whichever providers the configured models use, and `--check-deps` lists optional packages.
- Missing-key and missing-SDK errors are reported immediately instead of being retried.

## 1.1.0 — 2026-09-01

Google shut down the models this skill hard-coded (`gemini-2.0-flash` on 2026-06-01, `gemini-3-pro-image-preview` on 2026-06-25), which broke diagram mode. This release restores it and fixes bugs found along the way.

### Changed
- Default models are now `gemini-3.5-flash` (Retriever, Planner, Stylist, Critic) and `gemini-3-pro-image` (Visualizer). Override with `PAPERBANANA_VLM_MODEL` / `PAPERBANANA_IMAGE_MODEL`, or `--vlm-model` / `--image-model` on `orchestrate.py`. Model names live in one place: `scripts/common.py`.
- `GEMINI_API_KEY` is accepted alongside `GOOGLE_API_KEY`.
- Requirements: `google-genai>=2.0`, `matplotlib>=3.9`.
- SKILL.md frontmatter gains `license`, `compatibility`, and `metadata`.

### Fixed
- Aspect ratio is now actually sent to the image API, and a new `--image-size 1K|2K|4K` option controls resolution (default 2K).
- Critic refinement iterations no longer overwrite each other. Every iteration is kept in the work directory and the best-scoring one is copied to `--output`.
- `retriever.py` accepts `--references-dir` when run on its own.
- `validate_output.py --check-deps` reports the real `google-genai` version, and a new `--check-api` verifies the key and model access.
- Box plots use matplotlib directly (no seaborn deprecation warnings); the violin fallback no longer passes an invalid argument.
- Docs: corrected paper title, model table, and added links to the official PaperBanana repo and PaperBananaBench.

## 1.0.0 — 2026-02

Initial release.
