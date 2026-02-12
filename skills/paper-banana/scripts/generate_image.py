#!/usr/bin/env python3
"""Google GenAI image generation for PaperBanana methodology diagrams.

Uses gemini-3-pro-image-preview to generate publication-ready academic
illustrations from detailed textual descriptions.

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
    print("Error: Pillow package not installed.")
    print("Install with: pip install pillow")
    sys.exit(1)


ASPECT_RATIOS = {
    "16:9": "16:9",
    "3:2": "3:2",
    "1:1": "1:1",
    "21:9": "21:9",
    "9:16": "9:16",
    "2:3": "2:3",
}

DEFAULT_MODEL = "gemini-3-pro-image-preview"
DEFAULT_ASPECT_RATIO = "16:9"
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds


def get_api_key() -> str:
    """Get Google API key from environment."""
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Set it with: export GOOGLE_API_KEY='your-api-key'")
        sys.exit(1)
    return key


def build_prompt(description: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO) -> str:
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
    model: str = DEFAULT_MODEL,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    temperature: float = 1.0,
) -> str:
    """Generate an image using Google GenAI.

    Args:
        prompt: The full text prompt for image generation.
        output_path: Path to save the generated image.
        model: Model name to use.
        aspect_ratio: Aspect ratio string (e.g., "16:9").
        temperature: Generation temperature (default 1.0).

    Returns:
        Path to the saved image file.

    Raises:
        RuntimeError: If image generation fails after all retries.
    """
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)

    # Validate aspect ratio
    if aspect_ratio not in ASPECT_RATIOS:
        print(f"Warning: Unknown aspect ratio '{aspect_ratio}'. Using {DEFAULT_ASPECT_RATIO}.")
        aspect_ratio = DEFAULT_ASPECT_RATIO

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
                    response_modalities=["image", "text"],
                ),
            )

            # Extract image from response
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                        image_data = part.inline_data.data
                        image = Image.open(io.BytesIO(image_data))

                        # Save as PNG
                        if not output_path.lower().endswith(".png"):
                            output_path += ".png"
                        image.save(output_path, "PNG", quality=95)
                        print(f"Image saved to: {output_path}")
                        print(f"Dimensions: {image.size[0]}x{image.size[1]}")
                        return output_path

            raise RuntimeError("No image data in API response")

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
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--aspect-ratio", type=str, default=DEFAULT_ASPECT_RATIO,
                        choices=list(ASPECT_RATIOS.keys()),
                        help=f"Aspect ratio (default: {DEFAULT_ASPECT_RATIO})")
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
    full_prompt = build_prompt(description, args.aspect_ratio)

    print(f"Model: {args.model}")
    print(f"Aspect ratio: {args.aspect_ratio}")
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
        )
        print(f"Success: {result_path}")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
