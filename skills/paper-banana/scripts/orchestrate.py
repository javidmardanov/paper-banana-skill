#!/usr/bin/env python3
"""PaperBanana Pipeline Orchestrator — End-to-end multi-agent execution.

Chains the 5 PaperBanana agents sequentially:
  Retriever → Planner → Stylist → Visualizer → Critic
with the Critic's refinement loop (up to 3 iterations).

Supports both Diagram Mode (Gemini image generation) and Plot Mode
(matplotlib/seaborn code generation).

Usage:
    # Diagram mode
    python orchestrate.py --methodology methodology.txt \
        --caption "Figure 1: Overview of proposed framework" \
        --mode diagram --output output/diagram.png

    # Diagram mode with inline text
    python orchestrate.py --methodology "Our framework consists of..." \
        --caption "Figure 1: System overview" \
        --mode diagram --output output/diagram.png

    # Plot mode
    python orchestrate.py --data data.json \
        --intent "bar chart comparing model accuracy" \
        --mode plot --output output/figure.pdf

Requirements:
    pip install google-genai pillow matplotlib seaborn numpy
    export GOOGLE_API_KEY="your-api-key"
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import image_model, vlm_model
from retriever import run_retriever
from planner import run_planner
from stylist import run_stylist
from generate_image import generate_image, build_prompt
from critic import run_critic

SKILL_DIR = SCRIPT_DIR.parent
MAX_REFINEMENTS = 3


def determine_aspect_ratio(visual_intent: str) -> str:
    """Select aspect ratio based on visual intent."""
    mapping = {
        "Pipeline/Flow": "16:9",
        "Framework Overview": "16:9",
        "Detailed Module": "3:2",
        "Architecture Diagram": "3:2",
    }
    return mapping.get(visual_intent, "16:9")


def _primary_score(critic_output: dict) -> float:
    """Rank iterations by the Critic's primary dimensions (faithfulness + readability)."""
    scores = critic_output.get("scores", {})
    try:
        return float(scores.get("faithfulness", 0)) + float(scores.get("readability", 0))
    except (TypeError, ValueError):
        return 0.0


def save_intermediate(data: dict, name: str, work_dir: Path) -> Path:
    """Save intermediate JSON output to working directory."""
    path = work_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def run_diagram_pipeline(
    methodology: str,
    caption: str,
    output_path: str,
    work_dir: Path,
    references_dir: str = None,
    image_size: str = "2K",
) -> dict:
    """Run the full diagram generation pipeline.

    Args:
        methodology: The user's methodology text.
        caption: The figure caption.
        output_path: Final image output path.
        work_dir: Working directory for intermediate files.
        references_dir: Optional custom references directory.
        image_size: Visualizer output resolution ("1K", "2K", "4K").

    Returns:
        Dict with final results including scores and output path.
    """
    results = {
        "mode": "diagram",
        "iterations": [],
        "models": {"vlm": vlm_model(), "image": image_model()},
    }
    start_time = time.time()
    print(f"Models: VLM={vlm_model()}  image={image_model()}")

    # === Phase 1: Retriever ===
    print("\n" + "=" * 60)
    print("PHASE 1: RETRIEVER — Categorizing & selecting references")
    print("=" * 60)
    retriever_output = run_retriever(methodology, mode="diagram", references_dir=references_dir)
    save_intermediate(retriever_output, "retriever_output", work_dir)
    results["category"] = retriever_output.get("category", "")
    results["visual_intent"] = retriever_output.get("visual_intent", "")

    # === Phase 2: Planner ===
    print("\n" + "=" * 60)
    print("PHASE 2: PLANNER — Generating detailed description")
    print("=" * 60)
    planner_output = run_planner(methodology, caption, retriever_output)
    save_intermediate(planner_output, "planner_output", work_dir)

    # === Phase 3: Stylist ===
    print("\n" + "=" * 60)
    print("PHASE 3: STYLIST — Applying NeurIPS 2025 aesthetics")
    print("=" * 60)
    stylist_output = run_stylist(planner_output)
    save_intermediate(stylist_output, "stylist_output", work_dir)

    # === Phase 4 + 5: Visualizer + Critic Loop ===
    current_description = stylist_output["styled_description"]
    aspect_ratio = determine_aspect_ratio(results.get("visual_intent", ""))

    best = None  # (score, iteration, image_path, critic_output)

    for iteration in range(1, MAX_REFINEMENTS + 1):
        print("\n" + "=" * 60)
        print(f"PHASE 4: VISUALIZER — Generating image (iteration {iteration})")
        print("=" * 60)

        # Every iteration is kept in work_dir; the best one is copied to output_path.
        full_prompt = build_prompt(current_description)
        iter_output = str(work_dir / f"diagram_iter{iteration}.png")

        try:
            generate_image(
                prompt=full_prompt,
                output_path=iter_output,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
            )
        except RuntimeError as e:
            print(f"Error: Image generation failed: {e}")
            results["error"] = str(e)
            break

        print(f"\n{'=' * 60}")
        print(f"PHASE 5: CRITIC — Evaluating image (iteration {iteration})")
        print("=" * 60)

        critic_output = run_critic(
            image_path=iter_output,
            methodology=methodology,
            stylist_output=stylist_output,
            iteration=iteration,
        )
        critic_output["image_path"] = iter_output
        save_intermediate(critic_output, f"critic_output_iter{iteration}", work_dir)
        results["iterations"].append(critic_output)

        score = _primary_score(critic_output)
        if best is None or score > best[0]:
            best = (score, iteration, iter_output, critic_output)

        if critic_output.get("primary_pass", False):
            print(f"\n  Image ACCEPTED at iteration {iteration}")
            break

        if iteration >= MAX_REFINEMENTS:
            print(f"\n  Max refinements ({MAX_REFINEMENTS}) reached. Keeping best-scoring version.")
            results["max_refinements_reached"] = True
            break

        revised = critic_output.get("revised_description")
        if not revised:
            print("  No revised description provided. Keeping current version.")
            break
        print(f"\n  Revising description for iteration {iteration + 1}...")
        current_description = revised
        stylist_output["styled_description"] = revised
        save_intermediate(stylist_output, f"stylist_output_revised_iter{iteration}", work_dir)

    if best is not None:
        _, best_iter, best_path, best_critic = best
        shutil.copyfile(best_path, output_path)
        results["accepted"] = True
        results["best_iteration"] = best_iter
        results["final_scores"] = best_critic.get("scores", {})
        print(f"\n  Final image: iteration {best_iter} copied to {output_path}")

    elapsed = time.time() - start_time
    results["output_path"] = output_path
    results["elapsed_seconds"] = round(elapsed, 1)
    results["work_dir"] = str(work_dir)

    return results


def run_plot_pipeline(
    data_path: str,
    intent: str,
    output_path: str,
    work_dir: Path,
) -> dict:
    """Run the plot generation pipeline.

    Plot mode generates Python code rather than calling image generation.
    This is handled by the existing plot_generator.py or by Claude generating
    custom matplotlib code. The orchestrator provides the structure.

    Args:
        data_path: Path to data JSON file.
        intent: Description of desired plot type and purpose.
        output_path: Output file path for the generated plot.
        work_dir: Working directory for intermediate files.

    Returns:
        Dict with results.
    """
    print("\n" + "=" * 60)
    print("PLOT MODE")
    print("=" * 60)
    print("Plot mode uses code-based generation to eliminate data hallucination.")
    print(f"Data: {data_path}")
    print(f"Intent: {intent}")
    print(f"Output: {output_path}")
    print()
    print("For plot generation, use one of:")
    print(f"  1. python {SCRIPT_DIR / 'plot_generator.py'} --config {data_path} --output {output_path}")
    print("  2. Ask your AI agent to generate custom matplotlib/seaborn code")
    print()
    print("Plot mode does not use Gemini image generation — code-based generation")
    print("eliminates data hallucination errors that corrupt numerical accuracy.")

    # If data file exists and has the right structure, try plot_generator
    if data_path and Path(data_path).exists():
        try:
            from plot_generator import main as plot_main
            sys.argv = ["plot_generator.py", "--config", data_path, "--output", output_path]
            plot_main()
            return {
                "mode": "plot",
                "output_path": output_path,
                "method": "plot_generator",
            }
        except Exception as e:
            print(f"  plot_generator.py failed: {e}")
            print("  Generate custom matplotlib code instead.")

    return {
        "mode": "plot",
        "output_path": output_path,
        "method": "manual",
        "note": "Plot mode requires code generation. Use plot_generator.py with a JSON config or generate custom matplotlib code.",
    }


def print_summary(results: dict) -> None:
    """Print a final summary of the pipeline run."""
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Mode: {results.get('mode', 'unknown')}")
    print(f"  Output: {results.get('output_path', 'N/A')}")

    if results.get("mode") == "diagram":
        print(f"  Category: {results.get('category', 'N/A')}")
        print(f"  Visual intent: {results.get('visual_intent', 'N/A')}")
        print(f"  Iterations: {len(results.get('iterations', []))}")
        if results.get("best_iteration"):
            print(f"  Best iteration: {results['best_iteration']}")
        models = results.get("models", {})
        if models:
            print(f"  Models: VLM={models.get('vlm')}  image={models.get('image')}")
        print(f"  Accepted: {results.get('accepted', False)}")
        print(f"  Elapsed: {results.get('elapsed_seconds', 0)}s")

        scores = results.get("final_scores", {})
        if scores:
            print("  Final scores:")
            print(f"    Faithfulness: {scores.get('faithfulness', '?')}/10")
            print(f"    Readability:  {scores.get('readability', '?')}/10")
            print(f"    Conciseness:  {scores.get('conciseness', '?')}/10")
            print(f"    Aesthetics:   {scores.get('aesthetics', '?')}/10")

        if results.get("max_refinements_reached"):
            print("  Note: Max refinements reached. Manual review recommended.")

    print(f"  Intermediates: {results.get('work_dir', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="PaperBanana Pipeline Orchestrator — End-to-end multi-agent execution"
    )
    # Mode
    parser.add_argument("--mode", choices=["diagram", "plot"], default="diagram",
                        help="Output mode (default: diagram)")

    # Diagram mode inputs
    diag_group = parser.add_mutually_exclusive_group()
    diag_group.add_argument("--methodology", type=str,
                            help="Methodology text (diagram mode)")
    diag_group.add_argument("--methodology-file", type=str,
                            help="File containing methodology text (diagram mode)")
    parser.add_argument("--caption", type=str, default="",
                        help="Figure caption (diagram mode)")

    # Plot mode inputs
    parser.add_argument("--data", type=str,
                        help="Path to data JSON file (plot mode)")
    parser.add_argument("--intent", type=str, default="",
                        help="Plot intent description (plot mode)")

    # Common
    parser.add_argument("--output", type=str, default="output/diagram.png",
                        help="Output file path (default: output/diagram.png)")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Working directory for intermediates (default: output/work/)")
    parser.add_argument("--references-dir", type=str, default=None,
                        help="Custom references directory (must contain index.json + images)")
    parser.add_argument("--image-size", choices=["1K", "2K", "4K"], default="2K",
                        help="Diagram output resolution (default: 2K)")
    parser.add_argument("--vlm-model", type=str, default=None,
                        help=f"Gemini model for Retriever/Planner/Stylist/Critic (default: {vlm_model()})")
    parser.add_argument("--image-model", type=str, default=None,
                        help=f"Gemini image model for the Visualizer (default: {image_model()})")

    args = parser.parse_args()

    # CLI overrides win over PAPERBANANA_*_MODEL environment variables.
    if args.vlm_model:
        os.environ["PAPERBANANA_VLM_MODEL"] = args.vlm_model
    if args.image_model:
        os.environ["PAPERBANANA_IMAGE_MODEL"] = args.image_model

    # Set up working directory
    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        work_dir = Path(args.output).parent / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "diagram":
        # Get methodology text
        if args.methodology_file:
            path = Path(args.methodology_file)
            if not path.exists():
                print(f"Error: File not found: {args.methodology_file}")
                sys.exit(1)
            methodology = path.read_text(encoding="utf-8").strip()
        elif args.methodology:
            methodology = args.methodology
        else:
            print("Error: Diagram mode requires --methodology or --methodology-file")
            sys.exit(1)

        results = run_diagram_pipeline(methodology, args.caption, args.output, work_dir,
                                       args.references_dir, args.image_size)
    else:
        if not args.data:
            print("Error: Plot mode requires --data")
            sys.exit(1)
        results = run_plot_pipeline(args.data, args.intent, args.output, work_dir)

    # Save final results
    save_intermediate(results, "pipeline_results", work_dir)
    print_summary(results)


if __name__ == "__main__":
    main()
