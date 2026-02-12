#!/usr/bin/env python3
"""Output validation for PaperBanana generated code and images.

Validates Python syntax, checks dependencies, and verifies generated
output files exist and meet basic quality requirements.

Usage:
    python validate_output.py --check-deps
    python validate_output.py --check-code generated_plot.py
    python validate_output.py --check-image output/diagram.png
    python validate_output.py --run generated_plot.py --output output/figure.pdf
"""

import argparse
import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path


APPROVED_IMPORTS = {
    "matplotlib", "matplotlib.pyplot", "matplotlib.patches",
    "matplotlib.lines", "matplotlib.colors", "matplotlib.cm",
    "matplotlib.gridspec", "matplotlib.ticker", "matplotlib.font_manager",
    "seaborn",
    "numpy",
    "pandas",
    "math",
    "json",
    "os",
    "pathlib",
}

REQUIRED_PACKAGES = {
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "numpy": "numpy",
    "pillow": "PIL",
    "google-genai": "google.genai",
}


def check_dependencies() -> dict:
    """Check which required packages are installed.

    Returns:
        Dictionary mapping package names to (installed: bool, version: str | None).
    """
    results = {}
    for package_name, import_name in REQUIRED_PACKAGES.items():
        try:
            mod = importlib.import_module(import_name.split(".")[0])
            version = getattr(mod, "__version__", "unknown")
            results[package_name] = {"installed": True, "version": version}
        except ImportError:
            results[package_name] = {"installed": False, "version": None}
    return results


def check_code(code_path: str) -> dict:
    """Validate generated Python code.

    Checks:
    - Valid Python syntax
    - All imports from approved set
    - Contains savefig call
    - No plt.show() call
    - Has output path defined

    Args:
        code_path: Path to the Python file to validate.

    Returns:
        Dictionary with validation results.
    """
    path = Path(code_path)
    if not path.exists():
        return {"valid": False, "errors": [f"File not found: {code_path}"]}

    code = path.read_text(encoding="utf-8")
    errors = []
    warnings = []

    # Check syntax
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"valid": False, "errors": [f"Syntax error at line {e.lineno}: {e.msg}"]}

    # Check imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                if alias.name not in APPROVED_IMPORTS and root_module not in {
                    m.split(".")[0] for m in APPROVED_IMPORTS
                }:
                    warnings.append(f"Non-standard import: {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_module = node.module.split(".")[0]
                if node.module not in APPROVED_IMPORTS and root_module not in {
                    m.split(".")[0] for m in APPROVED_IMPORTS
                }:
                    warnings.append(f"Non-standard import: from {node.module}")

    # Check for savefig
    has_savefig = "savefig" in code
    if not has_savefig:
        errors.append("Missing savefig() call — output will not be saved")

    # Check for plt.show()
    has_show = "plt.show()" in code or ".show()" in code
    if has_show:
        warnings.append("Contains plt.show() — will block in non-interactive mode. Remove it.")

    # Check for output path
    has_output_path = "OUTPUT_PATH" in code or "output_path" in code or "savefig" in code
    if not has_output_path:
        warnings.append("No OUTPUT_PATH variable defined")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "has_savefig": has_savefig,
        "has_show": has_show,
    }


def check_image(image_path: str) -> dict:
    """Validate a generated image file.

    Checks:
    - File exists
    - File is not empty
    - Valid image format (can be opened by PIL)
    - Minimum resolution

    Args:
        image_path: Path to the image file to validate.

    Returns:
        Dictionary with validation results.
    """
    path = Path(image_path)
    if not path.exists():
        return {"valid": False, "errors": [f"File not found: {image_path}"]}

    file_size = path.stat().st_size
    if file_size == 0:
        return {"valid": False, "errors": ["File is empty"]}

    errors = []
    info = {"file_size_kb": round(file_size / 1024, 1)}

    try:
        from PIL import Image
        img = Image.open(path)
        info["width"] = img.size[0]
        info["height"] = img.size[1]
        info["format"] = img.format
        info["mode"] = img.mode

        # Check minimum resolution
        if img.size[0] < 300 or img.size[1] < 300:
            errors.append(f"Resolution too low: {img.size[0]}x{img.size[1]} (minimum 300x300)")

    except ImportError:
        info["note"] = "Pillow not installed, skipping image validation"
    except Exception as e:
        errors.append(f"Cannot open image: {e}")

    return {"valid": len(errors) == 0, "errors": errors, **info}


def run_code(code_path: str, output_path: str = None) -> dict:
    """Execute generated Python code and validate output.

    Args:
        code_path: Path to the Python script to execute.
        output_path: Expected output file path. If None, extracted from code.

    Returns:
        Dictionary with execution results.
    """
    path = Path(code_path)
    if not path.exists():
        return {"success": False, "errors": [f"File not found: {code_path}"]}

    # First validate the code
    validation = check_code(code_path)
    if not validation["valid"]:
        return {"success": False, "errors": validation["errors"], "validation": validation}

    # Execute
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(path.parent),
        )

        output = {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }

        # Check if output file was created
        if output_path:
            output_file = Path(output_path)
            if output_file.exists():
                output["output_file"] = str(output_file)
                output["output_size_kb"] = round(output_file.stat().st_size / 1024, 1)
            else:
                output["warnings"] = [f"Expected output file not found: {output_path}"]

        return output

    except subprocess.TimeoutExpired:
        return {"success": False, "errors": ["Script execution timed out (60s limit)"]}
    except Exception as e:
        return {"success": False, "errors": [f"Execution error: {e}"]}


def main():
    parser = argparse.ArgumentParser(
        description="Validate PaperBanana generated outputs"
    )
    parser.add_argument("--check-deps", action="store_true",
                        help="Check if all required packages are installed")
    parser.add_argument("--check-code", type=str,
                        help="Validate generated Python code")
    parser.add_argument("--check-image", type=str,
                        help="Validate generated image file")
    parser.add_argument("--run", type=str,
                        help="Execute Python script and validate output")
    parser.add_argument("--output", type=str,
                        help="Expected output file path (used with --run)")

    args = parser.parse_args()

    if not any([args.check_deps, args.check_code, args.check_image, args.run]):
        parser.print_help()
        sys.exit(1)

    if args.check_deps:
        print("Checking dependencies...")
        results = check_dependencies()
        all_installed = True
        for package, info in results.items():
            status = f"v{info['version']}" if info["installed"] else "NOT INSTALLED"
            marker = "+" if info["installed"] else "-"
            print(f"  [{marker}] {package}: {status}")
            if not info["installed"]:
                all_installed = False

        if not all_installed:
            print("\nInstall missing packages with:")
            missing = [p for p, i in results.items() if not i["installed"]]
            print(f"  pip install {' '.join(missing)}")
            sys.exit(1)
        else:
            print("\nAll dependencies installed.")

    if args.check_code:
        print(f"Validating code: {args.check_code}")
        result = check_code(args.check_code)
        if result["valid"]:
            print("  Code is valid.")
        else:
            print("  Errors:")
            for error in result["errors"]:
                print(f"    - {error}")
        if result.get("warnings"):
            print("  Warnings:")
            for warning in result["warnings"]:
                print(f"    - {warning}")

    if args.check_image:
        print(f"Validating image: {args.check_image}")
        result = check_image(args.check_image)
        if result["valid"]:
            print(f"  Valid image: {result.get('width')}x{result.get('height')} "
                  f"{result.get('format', 'unknown')} ({result.get('file_size_kb')}KB)")
        else:
            print("  Errors:")
            for error in result["errors"]:
                print(f"    - {error}")

    if args.run:
        print(f"Executing: {args.run}")
        result = run_code(args.run, args.output)
        if result["success"]:
            print("  Execution successful.")
            if result.get("stdout"):
                print(f"  Output: {result['stdout']}")
            if result.get("output_file"):
                print(f"  Generated: {result['output_file']} ({result['output_size_kb']}KB)")
        else:
            print("  Execution failed:")
            for error in result.get("errors", []):
                print(f"    - {error}")
            if result.get("stderr"):
                print(f"  stderr: {result['stderr']}")


if __name__ == "__main__":
    main()
