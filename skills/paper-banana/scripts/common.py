#!/usr/bin/env python3
"""Shared helpers for the PaperBanana scripts.

Model routing and API calls live in providers.py and are re-exported here so every
agent imports from one place:

    PAPERBANANA_VLM_MODEL    Retriever / Planner / Stylist / Critic   default gemini-3.5-flash
    PAPERBANANA_IMAGE_MODEL  Visualizer                               default gemini-3-pro-image

Use ``provider/model`` to switch providers, e.g. openai/gpt-5.5, anthropic/claude-opus-5,
openai/gpt-image-2, openrouter/google/gemini-3.5-flash. See providers.py for env vars.
"""

import json
import re
from pathlib import Path

from providers import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_VLM_MODEL,
    ProviderConfigError,
    ProviderError,
    chat,
    check_model,
    generate_image_bytes,
    image_model,
    parse_model,
    usage_summary,
    vlm_model,
)

__all__ = [
    "DEFAULT_IMAGE_MODEL", "DEFAULT_VLM_MODEL", "ProviderConfigError", "ProviderError",
    "chat", "check_model", "generate_image_bytes", "image_model", "parse_model",
    "usage_summary", "vlm_model", "strip_code_fences", "load_methodology", "parse_pages",
    "extract_pdf_text", "clean_latex", "load_venues", "load_venue", "venue_figure_width",
]

SKILL_DIR = Path(__file__).resolve().parent.parent
VENUES_PATH = SKILL_DIR / "assets" / "venues.json"


def strip_code_fences(text: str) -> str:
    """Remove ```json fences a model may wrap around JSON output."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


# ---------------------------------------------------------------------------
# Methodology input: plain text, Markdown, LaTeX, or PDF
# ---------------------------------------------------------------------------

def load_methodology(text: str = None, file: str = None, pages: str = None) -> str:
    """Return methodology text from inline text or a .txt/.md/.tex/.pdf file."""
    if text and text.strip():
        return text.strip()
    if not file:
        raise ValueError("Provide methodology text or a methodology file (.txt, .md, .tex, .pdf).")
    path = Path(file)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(path, pages)
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".tex":
        return clean_latex(raw)
    return raw.strip()


def parse_pages(spec: str, n_pages: int) -> list[int]:
    """'3-5,8' -> [2, 3, 4, 7] (0-based, clipped to the document)."""
    if not spec:
        return list(range(n_pages))
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, _, b = chunk.partition("-")
        start, end = int(a), int(b or a)
        out.extend(i - 1 for i in range(start, end + 1) if 1 <= i <= n_pages)
    return out


def extract_pdf_text(path: Path, pages: str = None) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ProviderError("PDF input requires pypdf: pip install pypdf") from e
    reader = PdfReader(str(path))
    chunks = [(reader.pages[i].extract_text() or "") for i in parse_pages(pages, len(reader.pages))]
    text = "\n\n".join(c.strip() for c in chunks if c.strip())
    if not text:
        raise ValueError(f"No extractable text in {path} (scanned PDF?).")
    return text


def clean_latex(text: str) -> str:
    """Strip LaTeX noise (comments, citations, refs, figure environments) but keep math."""
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\begin\{(figure|table|algorithm)\*?\}.*?\\end\{\1\*?\}", "", text, flags=re.S)
    text = re.sub(r"\\(?:cite[pt]?|ref|eqref|label|autoref|cref|Cref|footnote)\*?(?:\[[^\]]*\])?\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}", r"\n\1\n", text)
    text = re.sub(r"\\(?:textbf|textit|emph|texttt|textsc)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"~", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Venue style packs
# ---------------------------------------------------------------------------

def load_venues() -> dict:
    with open(VENUES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_venue(name: str = "neurips") -> dict:
    """Return the venue entry (typesetting widths + style notes) for a venue key."""
    venues = load_venues()
    key = (name or "neurips").lower()
    if key not in venues:
        raise ValueError(f"Unknown venue '{name}'. Options: {', '.join(sorted(venues))}")
    return {"key": key, **venues[key]}


def venue_figure_width(venue: dict, width: str = "double") -> float:
    """Width in inches for a 'single' (one column) or 'double' (full text width) figure."""
    return venue["column_width_in"] if width == "single" else venue["text_width_in"]
