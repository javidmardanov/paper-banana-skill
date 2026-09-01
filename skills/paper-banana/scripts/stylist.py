#!/usr/bin/env python3
"""Stylist Agent — VLM-based style application for PaperBanana pipeline.

Takes the Planner's description and category, applies NeurIPS 2025 aesthetic
guidelines via Gemini VLM, and returns the polished styled description.

Usage:
    python stylist.py --description planner_output.json --output stylist_output.json
    python stylist.py --description planner_output.json --category "Science & Applications" --output stylist_output.json

Requirements:
    pip install google-genai
    export GOOGLE_API_KEY="your-api-key"
"""

import argparse
import json
import sys
from pathlib import Path

from common import chat, load_venue, venue_figure_width

SCRIPT_DIR = Path(__file__).parent
SKILL_DIR = SCRIPT_DIR.parent
STYLE_GUIDE_PATH = SKILL_DIR / "references" / "DIAGRAM-STYLE-GUIDE.md"


def load_style_guide() -> str:
    """Load the diagram style guide."""
    if STYLE_GUIDE_PATH.exists():
        return STYLE_GUIDE_PATH.read_text(encoding="utf-8")
    print("Warning: Style guide not found, using built-in rules.")
    return ""


def build_stylist_prompt(description: str, category: str, style_guide: str, venue_block: str = "") -> str:
    """Build the Stylist agent prompt for Gemini."""
    return f"""You are the Stylist agent in the PaperBanana academic illustration pipeline.

Your task: Refine the Planner's diagram description below to ensure it meets publication aesthetics for the target venue (the NeurIPS 2025 style guide below is the baseline). Apply domain-specific styling based on the diagram category.

--- FIVE CRITICAL RULES ---
1. PRESERVE high-quality aesthetics: Ensure the description produces a visually polished, professional result.
2. INTERVENE MINIMALLY: Only modify what needs improvement. Do not restructure a well-planned layout.
3. RESPECT DOMAIN DIVERSITY: Apply category-appropriate conventions. Agent papers look different from Vision papers.
4. ENRICH VAGUE DETAILS: If the Planner said "a box," specify "a rounded rectangle with soft blue fill and thin gray border."
5. PRESERVE CONTENT: Never remove, rename, or restructure components from the Planner's description.

--- STYLE APPLICATIONS BY CATEGORY ---
- Agent & Reasoning: Add persona icons, thought bubble motifs, warm illustrated style, playful but professional.
- Vision & Perception: Emphasize spatial relationships, use image-like thumbnails, geometric precision.
- Generative & Learning: Highlight flow and transformation, gradient fills for latent spaces, clear generation direction.
- Science & Applications: Conservative, domain-appropriate, minimal decoration, let content speak.

--- COLOR GUIDANCE ---
- Use "Soft Tech & Scientific Pastels" palette: cream, pale blue, mint, pale lavender as background zones.
- Functional colors: blue/orange pairings for active vs inactive.
- Warm tones (soft coral, amber) for trainable parameters.
- Cool tones (steel blue, sage) for frozen parameters.
- Avoid primary reds, neon colors, or high-saturation fills.
- CRITICAL: All colors in natural language only. NEVER use hex codes, RGB values, or CSS color names.

--- NEURIPS 2025 STYLE GUIDE ---
{style_guide}

{venue_block}
--- DIAGRAM CATEGORY ---
{category}

--- PLANNER'S DESCRIPTION ---
{description}

--- OUTPUT ---
Output the complete polished description ONLY. No explanations, commentary, reasoning, or preamble. Just the improved description text as flowing prose that an image generation model can follow."""


def run_stylist(planner_output: dict, category_override: str = None,
                venue: str = "neurips", figure_width: str = "double") -> dict:
    """Run the Stylist agent via the configured VLM.

    Args:
        planner_output: Output from the Planner agent.
        category_override: Optional category override.
        venue: Venue style pack key from assets/venues.json.
        figure_width: "single" (one column) or "double" (full text width).

    Returns:
        Dict with the styled description and metadata.
    """
    description = planner_output["description"]
    category = category_override or planner_output.get("category", "Science & Applications")
    style_guide = load_style_guide()

    venue_info = load_venue(venue)
    width_in = venue_figure_width(venue_info, figure_width)
    span = "single column" if figure_width == "single" else "full text width"
    venue_block = (
        f"--- TARGET VENUE ---\n"
        f"{venue_info['name']}: the figure will be typeset {width_in:.2f} inches wide ({span}). "
        f"All labels must stay legible at that size.\n{venue_info['notes']}\n"
    )
    prompt = build_stylist_prompt(description, category, style_guide, venue_block)

    print(f"Stylist: Applying {category} style for {venue_info['name']}...")
    styled_description = chat(prompt, temperature=0.3).strip()

    result = {
        "styled_description": styled_description,
        "category": category,
        "visual_intent": planner_output.get("visual_intent", ""),
        "caption": planner_output.get("caption", ""),
        "original_description": description,
        "venue": venue_info["key"],
        "figure_width_in": width_in,
    }

    print(f"  Styled description length: {len(styled_description)} chars")
    print(f"  Change delta: {len(styled_description) - len(description):+d} chars")

    return result


def main():
    parser = argparse.ArgumentParser(description="PaperBanana Stylist Agent")
    parser.add_argument("--description", type=str, required=True,
                        help="Path to planner_output.json")
    parser.add_argument("--category", type=str, default=None,
                        help="Override category (default: from planner output)")
    parser.add_argument("--output", type=str, default="stylist_output.json",
                        help="Output JSON path")
    parser.add_argument("--venue", type=str, default="neurips",
                        help="Venue style pack: neurips, iclr, icml, cvpr, acl, aaai (default: neurips)")
    parser.add_argument("--figure-width", choices=["single", "double"], default="double",
                        help="Typeset width the figure targets (default: double)")

    args = parser.parse_args()

    desc_path = Path(args.description)
    if not desc_path.exists():
        print(f"Error: Description file not found: {args.description}")
        sys.exit(1)
    with open(desc_path, "r", encoding="utf-8") as f:
        planner_output = json.load(f)

    result = run_stylist(planner_output, args.category, args.venue, args.figure_width)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
