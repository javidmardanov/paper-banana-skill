#!/usr/bin/env python3
"""Google GenAI image generation for PaperBanana methodology diagrams.

Uses a Gemini image model (default: gemini-3-pro-image, "Nano Banana Pro") to
generate publication-ready academic illustrations from detailed textual
descriptions. Override the model with --model or PAPERBANANA_IMAGE_MODEL.

Usage:
    python generate_image.py --prompt "description" --output "diagram.png"
    python generate_image.py --prompt-file description.txt --output "diagram.png"

Requirements:
    pip install google-genai pillow
    export GOOGLE_API_KEY="your-api-key"
"""

import argparse
import os
import sys
import time
from pathlib import Path

from common import get_client, image_model
from google.genai import types

try:
    from PIL import Image
    import io
except ImportError:
    print("Error: Pillow package not installed.")
    print("Install with: pip install pillow")
    sys.exit(1)


# Aspect ratios and output sizes accepted by the Gemini image models.
ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
IMAGE_SIZES = ["1K", "2K", "4K"]

DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_IMAGE_SIZE = "2K"
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds


def build_prompt(description: str) -> str:
    """Build the full image generation prompt from a styled description."""
    quality_prefix = (
        "High-resolution academic illustration for a top-tier ML conference paper. "
        "Clean white or very light background. "
        "All text must be perfectly legible in clear sans-serif font. "
        "Professional publication quality. "
        "No watermarks, signatures, or decorative borders. "
        "No figure number or caption text within the image. "
    )
    return quality_prefix + description


def generate_image(
    prompt: str,
    output_path: str,
    model: str = None,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    temperature: float = 1.0,
    image_size: str = DEFAULT_IMAGE_SIZE,
) -> str:
    """Generate an image using Google GenAI.

    Args:
        prompt: The full text prompt for image generation.
        output_path: Path to save the generated image.
        model: Model name (default: $PAPERBANANA_IMAGE_MODEL or gemini-3-pro-image).
        aspect_ratio: Aspect ratio string (e.g., "16:9").
        temperature: Generation temperature (default 1.0).
        image_size: Output resolution "1K", "2K", or "4K" (default 2K).

    Returns:
        Path to the saved image file.

    Raises:
        RuntimeError: If image generation fails after all retries.
    """
    client = get_client()
    model = model or image_model()

    if aspect_ratio not in ASPECT_RATIOS:
        print(f"Warning: Unknown aspect ratio '{aspect_ratio}'. Using {DEFAULT_ASPECT_RATIO}.")
        aspect_ratio = DEFAULT_ASPECT_RATIO
    if image_size not in IMAGE_SIZES:
        print(f"Warning: Unknown image size '{image_size}'. Using {DEFAULT_IMAGE_SIZE}.")
        image_size = DEFAULT_IMAGE_SIZE

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Retry loop with exponential backoff
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_modalities=["IMAGE", "TEXT"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=image_size,
                    ),
                ),
            )

            # Extract image from response (keep any text so refusals are visible)
            text_parts = []
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if part.text:
                        text_parts.append(part.text)
                    if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                        image_data = part.inline_data.data
                        image = Image.open(io.BytesIO(image_data))

                        # Save as PNG
                        if not output_path.lower().endswith(".png"):
                            output_path += ".png"
                        image.save(output_path, "PNG")
                        print(f"Image saved to: {output_path}")
                        print(f"Dimensions: {image.size[0]}x{image.size[1]}")
                        return output_path

            detail = " ".join(text_parts).strip()
            raise RuntimeError(
                "No image data in API response" + (f": {detail[:300]}" if detail else "")
            )

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"Attempt {attempt + 1} failed: {e}")
                print(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"Image generation failed after {MAX_RETRIES} attempts. "
                    f"Last error: {last_error}"
                ) from last_error


def main():
    parser = argparse.ArgumentParser(
        description="Generate academic illustrations using Google GenAI"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str, help="Text description for image generation")
    group.add_argument("--prompt-file", type=str, help="File containing the text description")
    parser.add_argument("--output", type=str, default="output/diagram.png",
                        help="Output file path (default: output/diagram.png)")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Image model (default: {image_model()}; "
                             "set PAPERBANANA_IMAGE_MODEL to change)")
    parser.add_argument("--aspect-ratio", type=str, default=DEFAULT_ASPECT_RATIO,
                        choices=ASPECT_RATIOS,
                        help=f"Aspect ratio (default: {DEFAULT_ASPECT_RATIO})")
    parser.add_argument("--image-size", type=str, default=DEFAULT_IMAGE_SIZE,
                        choices=IMAGE_SIZES,
                        help=f"Output resolution (default: {DEFAULT_IMAGE_SIZE})")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Generation temperature (default: 1.0)")

    args = parser.parse_args()

    # Get prompt text
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            print(f"Error: Prompt file not found: {args.prompt_file}")
            sys.exit(1)
        description = prompt_path.read_text(encoding="utf-8").strip()
    else:
        description = args.prompt

    # Build full prompt with quality prefix
    full_prompt = build_prompt(description)

    print(f"Model: {args.model or image_model()}")
    print(f"Aspect ratio: {args.aspect_ratio}  Size: {args.image_size}")
    print(f"Output: {args.output}")
    print(f"Prompt length: {len(full_prompt)} chars")
    print("Generating image...")

    try:
        result_path = generate_image(
            prompt=full_prompt,
            output_path=args.output,
            model=args.model,
            aspect_ratio=args.aspect_ratio,
            temperature=args.temperature,
            image_size=args.image_size,
        )
        print(f"Success: {result_path}")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
