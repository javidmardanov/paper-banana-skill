#!/usr/bin/env python3
"""Provider adapters for the PaperBanana pipeline.

Two operations cover every agent:

    chat(prompt, images=[...])        -> text     (Retriever, Planner, Stylist, Critic)
    generate_image_bytes(prompt, ...) -> PNG data (Visualizer)

Models are named ``provider/model``. Unprefixed names are inferred from the
model family (gemini-* -> gemini, gpt-* -> openai, claude-* -> anthropic,
vendor/model -> openrouter).

    Provider    Env var(s)                          Chat  Image
    gemini      GOOGLE_API_KEY or GEMINI_API_KEY    yes   yes   (gemini-3-pro-image, gemini-3.1-flash-image)
    openai      OPENAI_API_KEY [, OPENAI_BASE_URL]  yes   yes   (gpt-image-2, gpt-image-1.5)
    anthropic   ANTHROPIC_API_KEY (or ant auth)     yes   no
    openrouter  OPENROUTER_API_KEY                  yes   yes   (any model with image output)

Only the SDK for the provider you use needs to be installed.
"""

import base64
import os
import threading
from pathlib import Path

DEFAULT_VLM_MODEL = "gemini-3.5-flash"
DEFAULT_IMAGE_MODEL = "gemini-3-pro-image"
PROVIDERS = ("gemini", "openai", "anthropic", "openrouter")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


class ProviderError(RuntimeError):
    """Raised for missing SDKs, missing keys, refusals, or empty responses."""


class ProviderConfigError(ProviderError):
    """Configuration problem (missing key or SDK, unknown provider). Not worth retrying."""


def vlm_model() -> str:
    """Model used by the text/vision reasoning agents."""
    return os.environ.get("PAPERBANANA_VLM_MODEL", DEFAULT_VLM_MODEL)


def image_model() -> str:
    """Model used by the Visualizer for image generation."""
    return os.environ.get("PAPERBANANA_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)


def parse_model(spec: str) -> tuple[str, str]:
    """Split 'provider/model' into (provider, model); infer the provider if omitted."""
    spec = spec.strip()
    head, sep, rest = spec.partition("/")
    if sep and head in PROVIDERS and rest:
        return head, rest
    name = spec.lower()
    if name.startswith(("gemini", "imagen")):
        return "gemini", spec
    if name.startswith(("gpt", "o1", "o3", "o4", "chatgpt", "dall-e")):
        return "openai", spec
    if name.startswith("claude"):
        return "anthropic", spec
    if "/" in spec:
        return "openrouter", spec
    raise ProviderConfigError(
        f"Cannot infer a provider for model '{spec}'. Prefix it: gemini/{spec}, "
        f"openai/{spec}, anthropic/{spec}, or openrouter/<vendor>/{spec}."
    )


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

_USAGE: list[dict] = []
_USAGE_LOCK = threading.Lock()


def _record(model: str, input_tokens=0, output_tokens=0, images=0) -> None:
    with _USAGE_LOCK:
        _USAGE.append({
            "model": model,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "images": int(images or 0),
        })


def usage_summary() -> list[dict]:
    """Aggregate recorded usage per model (tokens and generated images)."""
    totals: dict[str, dict] = {}
    with _USAGE_LOCK:
        for u in _USAGE:
            t = totals.setdefault(u["model"], {"model": u["model"], "calls": 0,
                                               "input_tokens": 0, "output_tokens": 0, "images": 0})
            t["calls"] += 1
            for k in ("input_tokens", "output_tokens", "images"):
                t[k] += u[k]
    return list(totals.values())


def reset_usage() -> None:
    with _USAGE_LOCK:
        _USAGE.clear()


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

_CLIENTS: dict[str, object] = {}


def _require_key(*names: str) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    raise ProviderConfigError(f"Set {' or '.join(names)} in your environment.")


def _import(module: str, pip_name: str):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as e:
        raise ProviderConfigError(f"The '{pip_name}' package is required for this provider: pip install {pip_name}") from e


def _client(provider: str):
    if provider in _CLIENTS:
        return _CLIENTS[provider]
    if provider == "gemini":
        genai = _import("google.genai", "google-genai>=2")
        client = genai.Client(api_key=_require_key("GOOGLE_API_KEY", "GEMINI_API_KEY"))
    elif provider == "openai":
        openai = _import("openai", "openai")
        client = openai.OpenAI(api_key=_require_key("OPENAI_API_KEY"))  # honors OPENAI_BASE_URL
    elif provider == "openrouter":
        openai = _import("openai", "openai")
        client = openai.OpenAI(
            api_key=_require_key("OPENROUTER_API_KEY"),
            base_url=os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
            default_headers={"HTTP-Referer": "https://github.com/javidmardanov/paper-banana-skill",
                             "X-Title": "Paper Banana Skill"},
        )
    elif provider == "anthropic":
        anthropic = _import("anthropic", "anthropic")
        client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY or an `ant auth login` profile
    else:
        raise ProviderConfigError(f"Unknown provider '{provider}'.")
    _CLIENTS[provider] = client
    return client


def _image_file(path) -> tuple[bytes, str]:
    p = Path(path)
    mime = MIME_BY_SUFFIX.get(p.suffix.lower())
    if not mime:
        raise ProviderError(f"Unsupported image type '{p.suffix}' (use PNG, JPEG, or WebP): {p}")
    return p.read_bytes(), mime


def _data_url(path) -> str:
    data, mime = _image_file(path)
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


# ---------------------------------------------------------------------------
# chat(): text (+ optional images) -> text
# ---------------------------------------------------------------------------

def chat(prompt: str, images=None, json_mode: bool = False, temperature: float = None,
         model: str = None, max_tokens: int = 16000) -> str:
    """Send a prompt (with optional image paths) to the configured VLM and return its text.

    ``temperature`` is applied on Gemini only: current OpenAI reasoning models and
    Claude 4.7+ reject sampling parameters.
    """
    spec = model or vlm_model()
    provider, name = parse_model(spec)
    images = [Path(p) for p in (images or [])]
    if provider == "gemini":
        return _gemini_chat(name, prompt, images, json_mode, temperature)
    if provider in ("openai", "openrouter"):
        return _openai_chat(provider, name, prompt, images, json_mode)
    return _anthropic_chat(name, prompt, images, max_tokens)


def _gemini_chat(name, prompt, images, json_mode, temperature):
    from google.genai import types
    client = _client("gemini")
    cfg = {}
    if temperature is not None:
        cfg["temperature"] = temperature
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    if images:
        parts = [types.Part.from_bytes(data=d, mime_type=m) for d, m in map(_image_file, images)]
        parts.append(types.Part.from_text(text=prompt))
        contents = types.Content(parts=parts, role="user")
    else:
        contents = prompt
    response = client.models.generate_content(model=name, contents=contents,
                                              config=types.GenerateContentConfig(**cfg))
    um = getattr(response, "usage_metadata", None)
    _record(name, getattr(um, "prompt_token_count", 0), getattr(um, "candidates_token_count", 0))
    if not response.text:
        raise ProviderError(f"{name} returned no text.")
    return response.text


def _openai_chat(provider, name, prompt, images, json_mode):
    client = _client(provider)
    content = [{"type": "image_url", "image_url": {"url": _data_url(p)}} for p in images]
    content.append({"type": "text", "text": prompt})
    kwargs = {}
    if json_mode and provider == "openai":
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(
        model=name, messages=[{"role": "user", "content": content}], **kwargs)
    u = getattr(response, "usage", None)
    _record(name, getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
    text = response.choices[0].message.content if response.choices else None
    if not text:
        raise ProviderError(f"{name} returned no text.")
    return text


def _anthropic_chat(name, prompt, images, max_tokens):
    client = _client("anthropic")
    content = []
    for p in images:
        data, mime = _image_file(p)
        content.append({"type": "image", "source": {"type": "base64", "media_type": mime,
                                                    "data": base64.b64encode(data).decode("ascii")}})
    content.append({"type": "text", "text": prompt})
    response = client.messages.create(model=name, max_tokens=max_tokens,
                                      messages=[{"role": "user", "content": content}])
    u = getattr(response, "usage", None)
    _record(name, getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0))
    if response.stop_reason == "refusal":
        raise ProviderError(f"{name} refused the request.")
    text = "".join(b.text for b in response.content if b.type == "text")
    if not text:
        raise ProviderError(f"{name} returned no text.")
    return text


# ---------------------------------------------------------------------------
# generate_image_bytes(): prompt (+ optional reference images) -> PNG bytes
# ---------------------------------------------------------------------------

def generate_image_bytes(prompt: str, aspect_ratio: str = "16:9", image_size: str = "2K",
                         reference_images=None, model: str = None, temperature: float = None) -> bytes:
    """Generate one image. ``reference_images`` switches to edit/refine mode where supported."""
    spec = model or image_model()
    provider, name = parse_model(spec)
    refs = [Path(p) for p in (reference_images or [])]
    if provider == "gemini":
        return _gemini_image(name, prompt, aspect_ratio, image_size, refs, temperature)
    if provider == "openai":
        return _openai_image(name, prompt, aspect_ratio, image_size, refs)
    if provider == "openrouter":
        return _openrouter_image(name, prompt, aspect_ratio, refs)
    raise ProviderConfigError("Anthropic models do not generate images. Set PAPERBANANA_IMAGE_MODEL to a "
                        "Gemini, OpenAI, or OpenRouter image model.")


def _gemini_image(name, prompt, aspect_ratio, image_size, refs, temperature):
    from google.genai import types
    client = _client("gemini")
    cfg = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
        image_config=types.ImageConfig(aspect_ratio=aspect_ratio, image_size=image_size),
        **({"temperature": temperature} if temperature is not None else {}),
    )
    if refs:
        parts = [types.Part.from_bytes(data=d, mime_type=m) for d, m in map(_image_file, refs)]
        parts.append(types.Part.from_text(text=prompt))
        contents = types.Content(parts=parts, role="user")
    else:
        contents = prompt
    response = client.models.generate_content(model=name, contents=contents, config=cfg)
    um = getattr(response, "usage_metadata", None)
    _record(name, getattr(um, "prompt_token_count", 0), getattr(um, "candidates_token_count", 0), images=1)
    texts = []
    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                return part.inline_data.data
            if part.text:
                texts.append(part.text)
    detail = " ".join(texts).strip()
    raise ProviderError("No image in response" + (f": {detail[:300]}" if detail else ""))


def openai_size(model: str, aspect_ratio: str, image_size: str) -> str:
    """Map (aspect ratio, 1K/2K/4K) onto an OpenAI size string.

    gpt-image-2 accepts arbitrary WIDTHxHEIGHT (multiples of 16, ratio within 1:3..3:1,
    max 3840x2160). Older gpt-image models only accept 1024x1024, 1536x1024, 1024x1536.
    """
    w, h = (int(x) for x in aspect_ratio.split(":"))
    if "gpt-image-2" in model:
        long_side = {"1K": 1536, "2K": 2048, "4K": 3840}.get(image_size, 2048)
        if w >= h:
            width, height = long_side, long_side * h / w
        else:
            width, height = long_side * w / h, long_side
        short = min(width, height)
        if short > 2160:
            width, height = width * 2160 / short, height * 2160 / short
        return f"{int(round(width / 16) * 16)}x{int(round(height / 16) * 16)}"
    if w > h:
        return "1536x1024"
    if h > w:
        return "1024x1536"
    return "1024x1024"


def _openai_image(name, prompt, aspect_ratio, image_size, refs):
    client = _client("openai")
    kwargs = {
        "model": name,
        "prompt": prompt,
        "size": openai_size(name, aspect_ratio, image_size),
        "quality": os.environ.get("PAPERBANANA_OPENAI_IMAGE_QUALITY", "high"),
        "n": 1,
    }
    if refs:
        files = [(p.name, *_image_file(p)) for p in refs]  # (filename, bytes, mime)
        response = client.images.edit(image=files if len(files) > 1 else files[0], **kwargs)
    else:
        response = client.images.generate(**kwargs)
    u = getattr(response, "usage", None)
    _record(name, getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0), images=1)
    if not response.data or not response.data[0].b64_json:
        raise ProviderError(f"{name} returned no image data.")
    return base64.b64decode(response.data[0].b64_json)


def _openrouter_image(name, prompt, aspect_ratio, refs):
    client = _client("openrouter")
    content = [{"type": "image_url", "image_url": {"url": _data_url(p)}} for p in refs]
    content.append({"type": "text", "text": prompt})
    response = client.chat.completions.create(
        model=name,
        messages=[{"role": "user", "content": content}],
        extra_body={"modalities": ["image", "text"], "image_config": {"aspect_ratio": aspect_ratio}},
    )
    u = getattr(response, "usage", None)
    _record(name, getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0), images=1)
    message = response.choices[0].message if response.choices else None
    images = (message.model_dump().get("images") or []) if message is not None else []
    for img in images:
        url = (img.get("image_url") or {}).get("url", "")
        if url.startswith("data:"):
            return base64.b64decode(url.split(",", 1)[1])
    detail = (message.content or "") if message is not None else ""
    raise ProviderError("No image in response" + (f": {detail[:300]}" if detail else ""))


# ---------------------------------------------------------------------------
# check_model(): used by validate_output.py --check-api
# ---------------------------------------------------------------------------

def check_model(spec: str) -> None:
    """Raise ProviderError if the key is missing or the model is not reachable."""
    provider, name = parse_model(spec)
    client = _client(provider)
    try:
        if provider == "gemini":
            client.models.get(model=name)
        elif provider == "openrouter":
            if not any(m.id == name for m in client.models.list()):
                raise ProviderError(f"'{name}' is not in the OpenRouter model list.")
        else:
            client.models.retrieve(name)
    except ProviderError:
        raise
    except Exception as e:  # noqa: BLE001 - normalize SDK-specific errors
        first = str(e).strip().splitlines()[0] if str(e).strip() else repr(e)
        raise ProviderError(first[:200]) from e
