"""Exercise every adapter against fake SDK clients (no network, no keys)."""
import base64
import io
import types as T

import pytest
from PIL import Image

import providers
from providers import ProviderError, chat, check_model, generate_image_bytes


def png_bytes():
    b = io.BytesIO(); Image.new("RGB", (32, 16), "white").save(b, "PNG"); return b.getvalue()


@pytest.fixture
def fake_clients(monkeypatch):
    monkeypatch.setattr(providers, "_CLIENTS", {})
    providers.reset_usage()
    return providers._CLIENTS


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "ref.png"; p.write_bytes(png_bytes()); return p


class Recorder:
    def __init__(self, response): self.response = response; self.kwargs = None
    def __call__(self, **kwargs): self.kwargs = kwargs; return self.response


def ns(**kw): return T.SimpleNamespace(**kw)


# ---- OpenAI / OpenRouter chat -------------------------------------------------

def test_openai_chat_builds_multimodal_message(fake_clients, img):
    create = Recorder(ns(choices=[ns(message=ns(content='{"ok": true}'))], usage=ns(prompt_tokens=7, completion_tokens=3)))
    fake_clients["openai"] = ns(chat=ns(completions=ns(create=create)))
    out = chat("Return JSON", images=[img], json_mode=True, temperature=0.2, model="openai/gpt-5.5")
    assert out == '{"ok": true}'
    k = create.kwargs
    assert k["model"] == "gpt-5.5" and "temperature" not in k
    assert k["response_format"] == {"type": "json_object"}
    content = k["messages"][0]["content"]
    assert content[0]["type"] == "image_url" and content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[-1] == {"type": "text", "text": "Return JSON"}
    assert providers.usage_summary()[0]["input_tokens"] == 7


def test_openrouter_chat_skips_response_format(fake_clients):
    create = Recorder(ns(choices=[ns(message=ns(content="hi"))], usage=None))
    fake_clients["openrouter"] = ns(chat=ns(completions=ns(create=create)))
    assert chat("x", json_mode=True, model="openrouter/google/gemini-3.5-flash") == "hi"
    assert "response_format" not in create.kwargs and create.kwargs["model"] == "google/gemini-3.5-flash"


# ---- Anthropic chat ------------------------------------------------------------

def test_anthropic_chat_and_refusal(fake_clients, img):
    create = Recorder(ns(stop_reason="end_turn", content=[ns(type="thinking", thinking=""), ns(type="text", text="desc")],
                         usage=ns(input_tokens=5, output_tokens=2)))
    fake_clients["anthropic"] = ns(messages=ns(create=create))
    assert chat("p", images=[img], model="anthropic/claude-opus-5") == "desc"
    k = create.kwargs
    assert k["model"] == "claude-opus-5" and k["max_tokens"] == 16000 and "temperature" not in k
    assert k["messages"][0]["content"][0]["source"]["media_type"] == "image/png"
    create.response = ns(stop_reason="refusal", content=[], usage=None)
    with pytest.raises(ProviderError, match="refused"):
        chat("p", model="claude-opus-5")


# ---- Gemini chat + image -------------------------------------------------------

def test_gemini_chat_and_image(fake_clients, img):
    gen = Recorder(ns(text='{"a":1}', usage_metadata=ns(prompt_token_count=4, candidates_token_count=2)))
    fake_clients["gemini"] = ns(models=ns(generate_content=gen))
    assert chat("p", images=[img], json_mode=True, temperature=0.2, model="gemini-3.5-flash") == '{"a":1}'
    cfg = gen.kwargs["config"]
    assert cfg.response_mime_type == "application/json" and cfg.temperature == 0.2
    assert len(gen.kwargs["contents"].parts) == 2

    data = png_bytes()
    gen.response = ns(candidates=[ns(content=ns(parts=[ns(text="here", inline_data=None),
                                                       ns(text=None, inline_data=ns(mime_type="image/png", data=data))]))],
                      usage_metadata=None)
    out = generate_image_bytes("draw", aspect_ratio="3:2", image_size="4K", reference_images=[img], model="gemini-3-pro-image")
    assert out == data
    cfg = gen.kwargs["config"]
    assert cfg.image_config.aspect_ratio == "3:2" and cfg.image_config.image_size == "4K"
    assert cfg.response_modalities == ["IMAGE", "TEXT"] and len(gen.kwargs["contents"].parts) == 2

    gen.response = ns(candidates=[ns(content=ns(parts=[ns(text="I cannot draw that", inline_data=None)]))], usage_metadata=None)
    with pytest.raises(ProviderError, match="cannot draw"):
        generate_image_bytes("draw", model="gemini-3-pro-image")


# ---- OpenAI images -------------------------------------------------------------

def test_openai_image_generate_and_edit(fake_clients, img, monkeypatch):
    monkeypatch.setenv("PAPERBANANA_OPENAI_IMAGE_QUALITY", "medium")
    data = png_bytes()
    resp = ns(data=[ns(b64_json=base64.b64encode(data).decode())], usage=ns(input_tokens=1, output_tokens=2))
    generate, edit = Recorder(resp), Recorder(resp)
    fake_clients["openai"] = ns(images=ns(generate=generate, edit=edit))
    assert generate_image_bytes("draw", "16:9", "2K", model="openai/gpt-image-2") == data
    assert generate.kwargs["size"] == "2048x1152" and generate.kwargs["quality"] == "medium" and generate.kwargs["n"] == 1
    assert generate_image_bytes("fix", "16:9", "1K", reference_images=[img], model="gpt-image-1.5") == data
    name, blob, mime = edit.kwargs["image"]
    assert name == "ref.png" and blob == png_bytes() and mime == "image/png" and edit.kwargs["size"] == "1536x1024"
    assert providers.usage_summary()[0]["images"] == 1 and providers.usage_summary()[1]["images"] == 1


# ---- OpenRouter images ---------------------------------------------------------

def test_openrouter_image(fake_clients, img):
    data = png_bytes()
    url = "data:image/png;base64," + base64.b64encode(data).decode()
    msg = ns(content=None, model_dump=lambda: {"images": [{"type": "image_url", "image_url": {"url": url}}]})
    create = Recorder(ns(choices=[ns(message=msg)], usage=None))
    fake_clients["openrouter"] = ns(chat=ns(completions=ns(create=create)))
    assert generate_image_bytes("draw", "21:9", reference_images=[img], model="openrouter/google/gemini-3-pro-image") == data
    body = create.kwargs["extra_body"]
    assert body["modalities"] == ["image", "text"] and body["image_config"] == {"aspect_ratio": "21:9"}
    assert create.kwargs["messages"][0]["content"][0]["type"] == "image_url"


def test_anthropic_cannot_generate_images(fake_clients):
    with pytest.raises(ProviderError, match="do not generate images"):
        generate_image_bytes("draw", model="anthropic/claude-opus-5")


# ---- check_model + key handling -------------------------------------------------

def test_check_model(fake_clients):
    fake_clients["openrouter"] = ns(models=ns(list=lambda: [ns(id="google/gemini-3.5-flash")]))
    check_model("openrouter/google/gemini-3.5-flash")
    with pytest.raises(ProviderError, match="not in the OpenRouter"):
        check_model("openrouter/x/y")

    def boom(*a, **k): raise RuntimeError("404 model not found\nmore")
    fake_clients["gemini"] = ns(models=ns(get=boom))
    with pytest.raises(ProviderError, match="^404 model not found$"):
        check_model("gemini-2.0-flash")


def test_missing_key_is_config_error(fake_clients, monkeypatch):
    for v in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(providers.ProviderConfigError, match="GOOGLE_API_KEY or GEMINI_API_KEY"):
        chat("x", model="gemini-3.5-flash")
    with pytest.raises(providers.ProviderConfigError, match="OPENAI_API_KEY"):
        chat("x", model="gpt-5.5")
