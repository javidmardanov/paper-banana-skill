#!/usr/bin/env python3
"""Retriever Agent — VLM-based reference selection for PaperBanana pipeline.

Loads the curated reference index, formats candidates, and asks Gemini VLM
to select the 2 most relevant reference examples for the user's methodology.

Usage:
    python retriever.py --methodology "source text..." --mode diagram --output retriever_output.json
    python retriever.py --methodology-file methodology.txt --mode diagram --output retriever_output.json

Requirements:
    pip install google-genai
    export GOOGLE_API_KEY="your-api-key"
"""

import argparse
import json
import sys
from pathlib import Path

from common import chat, load_methodology, strip_code_fences

SCRIPT_DIR = Path(__file__).parent
SKILL_DIR = SCRIPT_DIR.parent
REFERENCES_DIR = SKILL_DIR / "assets" / "references"
CATEGORIES_PATH = SKILL_DIR / "references" / "DIAGRAM-CATEGORIES.md"


def load_categories() -> str:
    """Load diagram categories text."""
    if CATEGORIES_PATH.exists():
        return CATEGORIES_PATH.read_text(encoding="utf-8")
    return ""


def format_candidates(index: list[dict]) -> str:
    """Format reference entries as a numbered candidate list for the VLM."""
    lines = []
    for i, entry in enumerate(index, 1):
        lines.append(
            f"{i}. ID: {entry['id']}\n"
            f"   Category: {entry['category']}\n"
            f"   Caption: {entry['caption']}"
        )
    return "\n\n".join(lines)


def build_retriever_prompt(methodology: str, candidates_text: str, categories_text: str) -> str:
    """Build the Retriever agent prompt for Gemini."""
    return f"""You are the Retriever agent in the PaperBanana academic illustration pipeline.

Your task:
1. Read the user's methodology text below.
2. Classify it into one of these 4 diagram categories:
   - Agent & Reasoning
   - Vision & Perception
   - Generative & Learning
   - Science & Applications
3. Identify the visual intent (Framework Overview, Pipeline/Flow, Detailed Module, or Architecture Diagram).
4. From the numbered reference candidates below, select the 2 most relevant examples that would best guide generating a methodology diagram for this text. Choose references whose visual structure and domain best match the user's methodology.

{categories_text}

--- REFERENCE CANDIDATES ---
{candidates_text}

--- USER METHODOLOGY TEXT ---
{methodology}

--- OUTPUT FORMAT ---
Respond with ONLY valid JSON, no markdown fences:
{{
  "category": "<one of the 4 categories>",
  "visual_intent": "<Framework Overview | Pipeline/Flow | Detailed Module | Architecture Diagram>",
  "domain_signals": ["keyword1", "keyword2", "keyword3"],
  "selected_references": [
    {{"id": "<reference_id>", "reason": "<why this reference is relevant>"}},
    {{"id": "<reference_id>", "reason": "<why this reference is relevant>"}}
  ]
}}"""


def run_retriever(methodology: str, mode: str, references_dir: str = None) -> dict:
    """Run the Retriever agent via Gemini VLM.

    Args:
        methodology: The user's methodology text.
        mode: "diagram" or "plot".
        references_dir: Optional custom references directory (must contain index.json + images).

    Returns:
        Dict with category, visual_intent, and selected_references.
    """
    if mode == "plot":
        # Plot mode doesn't need image reference selection
        return {
            "category": "plot",
            "visual_intent": "statistical_plot",
            "domain_signals": [],
            "selected_references": [],
        }

    # Use custom references dir if provided
    ref_dir = Path(references_dir) if references_dir else REFERENCES_DIR
    index_path = ref_dir / "index.json"
    if not index_path.exists():
        print(f"Error: Reference index not found at {index_path}")
        sys.exit(1)
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    categories_text = load_categories()
    candidates_text = format_candidates(index)
    prompt = build_retriever_prompt(methodology, candidates_text, categories_text)

    print("Retriever: Classifying methodology and selecting references...")
    result = json.loads(strip_code_fences(chat(prompt, json_mode=True, temperature=0.2)))

    # Enrich selected references with file paths and metadata from index
    index_lookup = {entry["id"]: entry for entry in index}
    enriched_refs = []
    for ref in result.get("selected_references", []):
        ref_id = ref["id"]
        if ref_id in index_lookup:
            entry = index_lookup[ref_id]
            enriched_refs.append({
                "id": ref_id,
                "file": str(ref_dir / entry["file"]),
                "caption": entry["caption"],
                "description": entry.get("description", entry["caption"]),
                "reason": ref.get("reason", ""),
            })
        else:
            print(f"  Warning: Reference '{ref_id}' not found in index, skipping.")

    result["selected_references"] = enriched_refs

    print(f"  Category: {result.get('category', 'unknown')}")
    print(f"  Visual intent: {result.get('visual_intent', 'unknown')}")
    for ref in enriched_refs:
        print(f"  Selected: {ref['id']} — {ref['reason'][:80]}")

    return result


def main():
    parser = argparse.ArgumentParser(description="PaperBanana Retriever Agent")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--methodology", type=str, help="Methodology text")
    group.add_argument("--methodology-file", type=str, help="Methodology file (.txt, .md, .tex, or .pdf)")
    parser.add_argument("--mode", choices=["diagram", "plot"], default="diagram",
                        help="Output mode (default: diagram)")
    parser.add_argument("--output", type=str, default="retriever_output.json",
                        help="Output JSON path (default: retriever_output.json)")
    parser.add_argument("--references-dir", type=str, default=None,
                        help="Custom references directory (must contain index.json + images)")

    args = parser.parse_args()

    try:
        methodology = load_methodology(args.methodology, args.methodology_file)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    result = run_retriever(methodology, args.mode, args.references_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
