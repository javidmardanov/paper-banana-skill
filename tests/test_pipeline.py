"""End-to-end orchestration with the provider layer mocked out."""
import io
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from PIL import Image

import critic
import generate_image
import orchestrate
import planner
import retriever
import stylist
from providers import ProviderConfigError

RETRIEVER_JSON = ('```json\n{"category":"Agent & Reasoning","visual_intent":"Pipeline/Flow","domain_signals":["a"],'
                  '"selected_references":[{"id":"2601.05110v1","reason":"r"},{"id":"bogus","reason":"x"}]}\n```')


class FakeVLM:
    """Scores are keyed off the image filename so parallel critics stay deterministic."""

    def __init__(self, passing_if):
        self.passing_if = passing_if
        self.calls = []

    def __call__(self, prompt, images=None, json_mode=False, temperature=None, **kw):
        self.calls.append({"prompt": prompt, "images": [str(p) for p in (images or [])], "json_mode": json_mode})
        if "Retriever agent" in prompt:
            return RETRIEVER_JSON
        if "Planner agent" in prompt:
            return "A left-to-right diagram ..."
        if "Stylist agent" in prompt:
            return "A polished left-to-right diagram ..."
        if "Critic agent" in prompt:
            name = Path(self.calls[-1]["images"][0]).name
            ok = self.passing_if(name)
            f, r = (8, 9) if ok else (5, 6)
            return json.dumps({"scores": {"faithfulness": f, "readability": r, "conciseness": 7, "aesthetics": 7},
                               "primary_pass": ok, "overall_pass": ok, "critic_suggestions": ["s"],
                               "revised_description": None if ok else f"Revised after {name}"})
        raise AssertionError(f"unexpected prompt: {prompt[:60]}")


class FakeImageGen:
    """Each call returns a PNG of a different width so files are distinguishable."""

    def __init__(self):
        self.calls = []
        self._n = 0
        self._lock = threading.Lock()

    def __call__(self, prompt, aspect_ratio="16:9", image_size="2K", reference_images=None, model=None, temperature=None):
        with self._lock:
            self._n += 1
            n = self._n
        self.calls.append({"aspect_ratio": aspect_ratio, "image_size": image_size,
                           "reference_images": list(reference_images or [])})
        b = io.BytesIO()
        Image.new("RGB", (300 + 10 * n, 300), "white").save(b, "PNG")
        return b.getvalue()


@pytest.fixture
def fakes(monkeypatch):
    def install(passing_if):
        vlm, gen = FakeVLM(passing_if), FakeImageGen()
        for mod in (retriever, planner, stylist, critic):
            monkeypatch.setattr(mod, "chat", vlm)
        monkeypatch.setattr(generate_image, "generate_image_bytes", gen)
        return vlm, gen
    return install


def test_candidates_pick_best_and_copy_to_output(fakes, tmp_path):
    vlm, gen = fakes(passing_if=lambda name: "_c2" in name)
    work = tmp_path / "work"; work.mkdir()
    out = tmp_path / "final.png"
    res = orchestrate.run_diagram_pipeline("Our agent framework ...", "Figure 1", str(out), work,
                                           image_size="1K", venue="icml", figure_width="single", num_candidates=2)
    assert res["accepted"] and res["best_iteration"] == 1 and res["venue"] == "icml"
    assert res["final_scores"]["faithfulness"] == 8
    c1, c2 = work / "diagram_iter1_c1.png", work / "diagram_iter1_c2.png"
    assert c1.exists() and c2.exists() and not (work / "diagram_iter2.png").exists()
    assert out.read_bytes() == c2.read_bytes() != c1.read_bytes()
    assert (work / "critic_output_iter1_c2.json").exists()
    assert {c["aspect_ratio"] for c in gen.calls} == {"16:9"} and {c["image_size"] for c in gen.calls} == {"1K"}
    # bogus reference id dropped; ACL/ICML width reached the Stylist prompt
    assert len(json.load(open(work / "retriever_output.json"))["selected_references"]) == 1
    stylist_prompt = next(c["prompt"] for c in vlm.calls if "Stylist agent" in c["prompt"])
    assert "3.25 inches" in stylist_prompt and "ICML" in stylist_prompt


def test_refinement_improve_mode_and_resume(fakes, tmp_path):
    vlm, gen = fakes(passing_if=lambda name: "iter2" in name)
    existing = tmp_path / "old_figure.png"
    Image.new("RGB", (200, 100), "gray").save(existing)
    work = tmp_path / "work"; work.mkdir()
    out = tmp_path / "final.png"
    res = orchestrate.run_diagram_pipeline("Our method ...", "Figure 2", str(out), work, input_image=str(existing))
    assert res["best_iteration"] == 2 and len(res["iterations"]) == 2
    assert out.read_bytes() == (work / "diagram_iter2.png").read_bytes()
    assert (work / "stylist_output_revised_iter1.json").exists()
    planner_call = next(c for c in vlm.calls if "Planner agent" in c["prompt"])
    assert planner_call["images"][-1] == str(existing) and "EXISTING FIGURE" in planner_call["prompt"]
    assert all(c["reference_images"] == [str(existing)] for c in gen.calls)

    # Resume: retriever/planner/stylist must not be called again.
    vlm2, gen2 = fakes(passing_if=lambda name: True)
    res2 = orchestrate.run_diagram_pipeline("Our method ...", "Figure 2", str(out), work, resume=True)
    assert res2["accepted"] and len(gen2.calls) == 1
    assert all("Critic agent" in c["prompt"] for c in vlm2.calls)


def test_generation_failure_is_reported(fakes, tmp_path, monkeypatch):
    fakes(passing_if=lambda name: True)
    def broken(*a, **k): raise ProviderConfigError("Set OPENAI_API_KEY in your environment.")
    monkeypatch.setattr(generate_image, "generate_image_bytes", broken)
    monkeypatch.setattr(generate_image.time, "sleep", lambda s: pytest.fail("config errors must not be retried"))
    work = tmp_path / "work"; work.mkdir()
    res = orchestrate.run_diagram_pipeline("m", "c", str(tmp_path / "o.png"), work)
    assert "OPENAI_API_KEY" in res["error"] and not res.get("accepted")


def test_stylist_venue_block(fakes):
    vlm, _ = fakes(passing_if=lambda name: True)
    res = stylist.run_stylist({"description": "d", "category": "Vision & Perception"}, venue="acl", figure_width="single")
    assert res["venue"] == "acl" and res["figure_width_in"] == 3.03
    assert "3.03 inches" in vlm.calls[-1]["prompt"] and "single column" in vlm.calls[-1]["prompt"]


@pytest.mark.parametrize("script,flags", [
    ("orchestrate.py", ["--num-candidates", "--input-image", "--venue", "--resume", "--pages", "--vlm-model", "--image-model"]),
    ("generate_image.py", ["--reference-image", "--image-size"]),
    ("retriever.py", ["--references-dir"]),
    ("plot_generator.py", ["--venue", "--width"]),
    ("validate_output.py", ["--check-api"]),
])
def test_cli_help(script, flags):
    scripts = Path(__file__).resolve().parents[1] / "skills" / "paper-banana" / "scripts"
    out = subprocess.run([sys.executable, str(scripts / script), "--help"], capture_output=True, text=True, check=True).stdout
    for f in flags:
        assert f in out
