#!/usr/bin/env python3
"""Planner Agent — Multimodal in-context description generation for PaperBanana.

Takes methodology text, caption, and selected reference images from the Retriever,
sends a multimodal prompt to Gemini VLM with reference images as visual context,
and generates a detailed textual description of the target methodology diagram.

Usage:
    python planner.py --methodology "source text..." --caption "Figure 1: ..." \
        --references retriever_output.json --output planner_output.json

Requirements:
    pip install google-genai pillow
    export GOOGLE_API_KEY="your-api-key"
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai package not installed.")
    print("Install with: pip install google-genai")
    sys.exit(1)

try:
    from PIL import Image
    import io
except ImportError:
    print("Error: Pillow not installed. Install with: pip install pillow")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
SKILL_DIR = SCRIPT_DIR.parent

VLM_MODEL = "gemini-2.0-flash"


def get_api_key() -> str:
    """Get Google API key from environment."""
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)
    return key


def load_image_bytes(image_path: str) -> bytes:
    """Load an image file and return its bytes."""
    with open(image_path, "rb") as f:
        return f.read()


def build_planner_prompt(methodology: str, caption: str, category: str, visual_intent: str) -> str:
    """Build the text portion of the Planner prompt."""
    return f"""You are the Planner agent in the PaperBanana academic illustration pipeline.

Your task: Convert the methodology text and figure caption below into an extremely detailed textual description of a methodology diagram. This description will be fed directly to an image generation model.

The reference images provided above show examples of high-quality NeurIPS 2025 methodology diagrams. Use them as visual guides for layout, style, and detail level. Generate a description that would produce a diagram of similar quality.

Category: {category}
Visual Intent: {visual_intent}

--- CRITICAL RULES ---
1. Be MAXIMALLY specific. Vague specifications produce worse figures.
2. Use ONLY natural language for visual attributes. NEVER use hex codes (#E6F3FF), RGB values, or pixel dimensions. Image generation models render these as garbled text.
3. Every component must have: a name, a shape description, a relative position, and a relative size.
4. Every connection must have: a source, a target, a type (solid arrow, dashed arrow, bidirectional), and optionally a label.

--- YOUR DESCRIPTION MUST SPECIFY ---

Layout: Overall direction (left-to-right, top-to-bottom, radial), grid structure, spacing.

Components (for each): Exact label text, shape (rounded rectangle, circle, diamond, etc.), fill color in natural language (e.g., "soft blue", "pale mint"), border style, relative size.

Connections (for each): Source → Target, arrow style (solid filled, dashed open, thick flow), direction, label text, color.

Groupings: Bounding boxes or regions, background colors, group labels.

Annotations: Mathematical formulas as text, step numbers, input/output indicators.

--- FIGURE CAPTION ---
{caption}

--- METHODOLOGY TEXT ---
{methodology}

--- OUTPUT ---
Write a single, complete textual description as flowing descriptive prose. No bullet points, no JSON, no code. Just the description that an image generation model can follow to produce the diagram."""


def run_planner(methodology: str, caption: str, references_data: dict) -> dict:
    """Run the Planner agent via Gemini VLM with multimodal context.

    Args:
        methodology: The user's methodology text.
        caption: The figure caption.
        references_data: Output from the Retriever agent.

    Returns:
        Dict with the detailed description and metadata.
    """
    category = references_data.get("category", "Science & Applications")
    visual_intent = references_data.get("visual_intent", "Pipeline/Flow")
    selected_refs = references_data.get("selected_references", [])

    api_key = get_api_key()
    client = genai.Client(api_key=api_key)

    # Build multimodal content parts
    content_parts = []

    # Add reference images as visual in-context examples
    for ref in selected_refs:
        ref_path = ref.get("file", "")
        if ref_path and Path(ref_path).exists():
            print(f"Planner: Loading reference image: {ref['id']}")
            img_bytes = load_image_bytes(ref_path)
            ext = Path(ref_path).suffix.lower()
            mime_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
            content_parts.append(
                types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
            )
            content_parts.append(
                types.Part.from_text(
                    text=f"Reference example ({ref['id']}): {ref['caption']}"
                )
            )
        else:
            print(f"  Warning: Reference image not found: {ref_path}")

    # Add the main text prompt
    prompt_text = build_planner_prompt(methodology, caption, category, visual_intent)
    content_parts.append(types.Part.from_text(text=prompt_text))

    print("Planner: Generating detailed diagram description with multimodal context...")
    response = client.models.generate_content(
        model=VLM_MODEL,
        contents=types.Content(parts=content_parts, role="user"),
        config=types.GenerateContentConfig(
            temperature=0.4,
        ),
    )

    description = response.text.strip()

    result = {
        "description": description,
        "category": category,
        "visual_intent": visual_intent,
        "caption": caption,
        "reference_ids": [ref["id"] for ref in selected_refs],
    }

    print(f"  Description length: {len(description)} chars")
    print(f"  Category: {category}")
    print(f"  Visual intent: {visual_intent}")

    return result


def main():
    parser = argparse.ArgumentParser(description="PaperBanana Planner Agent")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--methodology", type=str, help="Methodology text")
    group.add_argument("--methodology-file", type=str, help="File containing methodology text")
    parser.add_argument("--caption", type=str, default="",
                        help="Figure caption")
    parser.add_argument("--references", type=str, required=True,
                        help="Path to retriever_output.json")
    parser.add_argument("--output", type=str, default="planner_output.json",
                        help="Output JSON path")

    args = parser.parse_args()

    if args.methodology_file:
        path = Path(args.methodology_file)
        if not path.exists():
            print(f"Error: File not found: {args.methodology_file}")
            sys.exit(1)
        methodology = path.read_text(encoding="utf-8").strip()
    else:
        methodology = args.methodology

    ref_path = Path(args.references)
    if not ref_path.exists():
        print(f"Error: References file not found: {args.references}")
        sys.exit(1)
    with open(ref_path, "r", encoding="utf-8") as f:
        references_data = json.load(f)

    result = run_planner(methodology, args.caption, references_data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
