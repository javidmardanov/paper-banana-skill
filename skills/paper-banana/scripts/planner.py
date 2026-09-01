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
import sys
from pathlib import Path

from common import chat, load_methodology

SCRIPT_DIR = Path(__file__).parent
SKILL_DIR = SCRIPT_DIR.parent


def build_planner_prompt(methodology: str, caption: str, category: str, visual_intent: str,
                         reference_notes: str = "", existing_figure: bool = False) -> str:
    """Build the text portion of the Planner prompt."""
    existing_note = ""
    if existing_figure:
        existing_note = (
            "\nThe LAST image above is the author's EXISTING FIGURE. Your description must "
            "preserve every component, label, and relationship it shows (corrected against the "
            "methodology text where they disagree) while improving layout, legibility, and "
            "aesthetics. Describe the improved figure, not the original.\n"
        )
    return f"""You are the Planner agent in the PaperBanana academic illustration pipeline.

Your task: Convert the methodology text and figure caption below into an extremely detailed textual description of a methodology diagram. This description will be fed directly to an image generation model.

The reference images provided above show examples of high-quality methodology diagrams from top ML venues. Use them as visual guides for layout, style, and detail level. Generate a description that would produce a diagram of similar quality.
{reference_notes}{existing_note}

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


def run_planner(methodology: str, caption: str, references_data: dict, input_image: str = None) -> dict:
    """Run the Planner agent via the configured VLM with multimodal context.

    Args:
        methodology: The user's methodology text.
        caption: The figure caption.
        references_data: Output from the Retriever agent.
        input_image: Optional existing figure to improve (attached after the references).

    Returns:
        Dict with the detailed description and metadata.
    """
    category = references_data.get("category", "Science & Applications")
    visual_intent = references_data.get("visual_intent", "Pipeline/Flow")
    selected_refs = references_data.get("selected_references", [])

    # Reference images are attached in order; the notes tell the model which is which.
    images = []
    notes = []
    for ref in selected_refs:
        ref_path = ref.get("file", "")
        if ref_path and Path(ref_path).exists():
            print(f"Planner: Loading reference image: {ref['id']}")
            images.append(ref_path)
            notes.append(f"Reference image {len(images)} ({ref['id']}): {ref['caption']}")
        else:
            print(f"  Warning: Reference image not found: {ref_path}")
    if input_image:
        if not Path(input_image).exists():
            raise FileNotFoundError(f"Input image not found: {input_image}")
        print(f"Planner: Loading existing figure to improve: {input_image}")
        images.append(input_image)

    prompt_text = build_planner_prompt(methodology, caption, category, visual_intent,
                                       reference_notes="\n".join(notes),
                                       existing_figure=bool(input_image))

    print("Planner: Generating detailed diagram description with multimodal context...")
    description = chat(prompt_text, images=images, temperature=0.4).strip()

    result = {
        "description": description,
        "category": category,
        "visual_intent": visual_intent,
        "caption": caption,
        "reference_ids": [ref["id"] for ref in selected_refs],
        "input_image": input_image,
    }

    print(f"  Description length: {len(description)} chars")
    print(f"  Category: {category}")
    print(f"  Visual intent: {visual_intent}")

    return result


def main():
    parser = argparse.ArgumentParser(description="PaperBanana Planner Agent")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--methodology", type=str, help="Methodology text")
    group.add_argument("--methodology-file", type=str, help="Methodology file (.txt, .md, .tex, or .pdf)")
    parser.add_argument("--caption", type=str, default="",
                        help="Figure caption")
    parser.add_argument("--references", type=str, required=True,
                        help="Path to retriever_output.json")
    parser.add_argument("--output", type=str, default="planner_output.json",
                        help="Output JSON path")
    parser.add_argument("--input-image", type=str, default=None,
                        help="Existing figure to improve (attached to the prompt after the references)")

    args = parser.parse_args()

    try:
        methodology = load_methodology(args.methodology, args.methodology_file)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    ref_path = Path(args.references)
    if not ref_path.exists():
        print(f"Error: References file not found: {args.references}")
        sys.exit(1)
    with open(ref_path, "r", encoding="utf-8") as f:
        references_data = json.load(f)

    result = run_planner(methodology, args.caption, references_data, args.input_image)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
