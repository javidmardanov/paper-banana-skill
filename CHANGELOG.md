# Changelog

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
