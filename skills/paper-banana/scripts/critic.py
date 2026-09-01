#!/usr/bin/env python3
"""Critic Agent — VLM-based image evaluation for PaperBanana pipeline.

Takes a generated image, the original methodology text, and the styled description,
sends them to Gemini VLM for multimodal evaluation on 4 dimensions, and returns
structured JSON scores with optional revised description for refinement.

Usage:
    python critic.py --image output/diagram.png --methodology "source text..." \
        --description stylist_output.json --output critic_output.json

Requirements:
    pip install google-genai pillow
    export GOOGLE_API_KEY="your-api-key"
"""

import argparse
import json
import sys
from pathlib import Path

from common import chat, load_methodology, strip_code_fences

SCRIPT_DIR = Path(__file__).parent
SKILL_DIR = SCRIPT_DIR.parent
RUBRIC_PATH = SKILL_DIR / "references" / "EVALUATION-RUBRIC.md"


def load_rubric() -> str:
    """Load the evaluation rubric."""
    if RUBRIC_PATH.exists():
        return RUBRIC_PATH.read_text(encoding="utf-8")
    return ""


def build_critic_prompt(methodology: str, styled_description: str, caption: str, rubric: str) -> str:
    """Build the text portion of the Critic evaluation prompt."""
    return f"""You are the Critic agent in the PaperBanana academic illustration pipeline.

Your task: Evaluate the generated methodology diagram image (provided above) against the original methodology text and styled description. Score it on 4 dimensions and determine whether revision is needed.

--- EVALUATION RUBRIC ---
{rubric}

--- SCORING DIMENSIONS ---
1. FAITHFULNESS (Primary, must >= 7): Does the image accurately represent every component, connection, and relationship described in the methodology? No hallucinated or missing elements.
2. READABILITY (Primary, must >= 7): Are all text labels legible? No overlapping components? Clear visual flow direction? Sufficient contrast?
3. CONCISENESS (Secondary): Good signal-to-noise ratio? Appropriate detail level? Adequate white space?
4. AESTHETICS (Secondary): Color harmony? Consistent style? Professional polish? Domain-appropriate?

--- ORIGINAL METHODOLOGY TEXT ---
{methodology}

--- FIGURE CAPTION ---
{caption}

--- STYLED DESCRIPTION (what the image should depict) ---
{styled_description}

--- EVALUATION INSTRUCTIONS ---
1. Carefully examine the generated image.
2. Compare every element against the methodology text and styled description.
3. Score each dimension from 1-10.
4. If faithfulness < 7 OR readability < 7, provide a revised_description that fixes ALL identified issues. The revised description must be a complete, standalone description (not a diff or patch).
5. If all primary scores >= 7, set revised_description to null.

--- OUTPUT FORMAT ---
Respond with ONLY valid JSON, no markdown fences:
{{
  "scores": {{
    "faithfulness": <1-10>,
    "readability": <1-10>,
    "conciseness": <1-10>,
    "aesthetics": <1-10>
  }},
  "primary_pass": <true if faithfulness >= 7 AND readability >= 7>,
  "overall_pass": <true if primary_pass AND conciseness >= 5 AND aesthetics >= 5>,
  "critic_suggestions": [
    "Specific actionable suggestion 1",
    "Specific actionable suggestion 2"
  ],
  "revised_description": "<complete improved description if revision needed, null if acceptable>"
}}"""


def run_critic(
    image_path: str,
    methodology: str,
    stylist_output: dict,
    iteration: int = 1,
) -> dict:
    """Run the Critic agent via Gemini VLM with multimodal evaluation.

    Args:
        image_path: Path to the generated diagram image.
        methodology: The original methodology text.
        stylist_output: Output from the Stylist agent.
        iteration: Current refinement iteration number.

    Returns:
        Dict with scores, pass/fail, suggestions, and optional revised_description.
    """
    styled_description = stylist_output.get("styled_description", "")
    caption = stylist_output.get("caption", "")
    rubric = load_rubric()

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Critic: Evaluating image (iteration {iteration})...")
    prompt_text = build_critic_prompt(methodology, styled_description, caption, rubric)
    result = json.loads(strip_code_fences(
        chat(prompt_text, images=[str(img_path)], json_mode=True, temperature=0.2)
    ))
    result["iteration"] = iteration

    # Print evaluation summary
    scores = result.get("scores", {})
    print(f"  Faithfulness: {scores.get('faithfulness', '?')}/10")
    print(f"  Readability:  {scores.get('readability', '?')}/10")
    print(f"  Conciseness:  {scores.get('conciseness', '?')}/10")
    print(f"  Aesthetics:   {scores.get('aesthetics', '?')}/10")
    print(f"  Primary pass: {result.get('primary_pass', False)}")
    print(f"  Overall pass: {result.get('overall_pass', False)}")

    if result.get("revised_description"):
        print("  Revision: Required — revised description generated")
    else:
        print("  Revision: Not needed — image accepted")

    for suggestion in result.get("critic_suggestions", []):
        print(f"  Suggestion: {suggestion[:100]}")

    return result


def main():
    parser = argparse.ArgumentParser(description="PaperBanana Critic Agent")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the generated diagram image")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--methodology", type=str, help="Methodology text")
    group.add_argument("--methodology-file", type=str, help="Methodology file (.txt, .md, .tex, or .pdf)")
    parser.add_argument("--description", type=str, required=True,
                        help="Path to stylist_output.json")
    parser.add_argument("--iteration", type=int, default=1,
                        help="Refinement iteration number")
    parser.add_argument("--output", type=str, default="critic_output.json",
                        help="Output JSON path")

    args = parser.parse_args()

    try:
        methodology = load_methodology(args.methodology, args.methodology_file)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    desc_path = Path(args.description)
    if not desc_path.exists():
        print(f"Error: Description file not found: {args.description}")
        sys.exit(1)
    with open(desc_path, "r", encoding="utf-8") as f:
        stylist_output = json.load(f)

    result = run_critic(args.image, methodology, stylist_output, args.iteration)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
