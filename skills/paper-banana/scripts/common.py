#!/usr/bin/env python3
"""Shared configuration for the PaperBanana scripts.

Model names live here so a single change (or environment variable) updates
every agent. Google retires Gemini model IDs on a schedule, so prefer GA IDs:

    PAPERBANANA_VLM_MODEL    Retriever / Planner / Stylist / Critic
                             default: gemini-3.5-flash
                             cheaper: gemini-3.1-flash-lite
    PAPERBANANA_IMAGE_MODEL  Visualizer (image generation)
                             default: gemini-3-pro-image      ("Nano Banana Pro")
                             cheaper: gemini-3.1-flash-image  ("Nano Banana 2")

The API key is read from GOOGLE_API_KEY or GEMINI_API_KEY.
"""

import os
import sys

try:
    from google import genai
except ImportError:
    print("Error: google-genai package not installed.")
    print("Install with: pip install 'google-genai>=2'")
    sys.exit(1)

DEFAULT_VLM_MODEL = "gemini-3.5-flash"
DEFAULT_IMAGE_MODEL = "gemini-3-pro-image"


def vlm_model() -> str:
    """Model used by the text/vision reasoning agents."""
    return os.environ.get("PAPERBANANA_VLM_MODEL", DEFAULT_VLM_MODEL)


def image_model() -> str:
    """Model used by the Visualizer for image generation."""
    return os.environ.get("PAPERBANANA_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)


def get_api_key() -> str:
    """Return the Gemini API key from the environment, or exit with guidance."""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("Error: GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable not set.")
        print("Get a key at https://aistudio.google.com/apikey and run:")
        print("  export GOOGLE_API_KEY='your-api-key'")
        sys.exit(1)
    return key


def get_client() -> "genai.Client":
    """Create a Gemini API client."""
    return genai.Client(api_key=get_api_key())


def strip_code_fences(text: str) -> str:
    """Remove ```json fences a model may wrap around JSON output."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text
