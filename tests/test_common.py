import pytest

import common
import providers
from providers import ProviderConfigError, openai_size, parse_model


@pytest.mark.parametrize("spec,expected", [
    ("gemini-3.5-flash", ("gemini", "gemini-3.5-flash")),
    ("gemini/gemini-3-pro-image", ("gemini", "gemini-3-pro-image")),
    ("gpt-image-2", ("openai", "gpt-image-2")),
    ("openai/gpt-5.5", ("openai", "gpt-5.5")),
    ("claude-opus-5", ("anthropic", "claude-opus-5")),
    ("openrouter/google/gemini-3.5-flash", ("openrouter", "google/gemini-3.5-flash")),
    ("black-forest-labs/flux.2-pro", ("openrouter", "black-forest-labs/flux.2-pro")),
])
def test_parse_model(spec, expected):
    assert parse_model(spec) == expected


def test_parse_model_unknown():
    with pytest.raises(ProviderConfigError):
        parse_model("mystery-model")


@pytest.mark.parametrize("model,ratio,size,expected", [
    ("gpt-image-2", "16:9", "2K", "2048x1152"),
    ("gpt-image-2", "16:9", "4K", "3840x2160"),
    ("gpt-image-2", "3:2", "4K", "3232x2160"),   # short side capped at 2160, multiples of 16
    ("gpt-image-2", "9:16", "1K", "864x1536"),
    ("gpt-image-2", "1:1", "1K", "1536x1536"),
    ("gpt-image-1.5", "16:9", "2K", "1536x1024"),
    ("gpt-image-1.5", "1:1", "2K", "1024x1024"),
    ("gpt-image-1", "2:3", "1K", "1024x1536"),
])
def test_openai_size(model, ratio, size, expected):
    assert openai_size(model, ratio, size) == expected
    w, h = map(int, expected.split("x"))
    if "gpt-image-2" in model:
        assert w % 16 == 0 and h % 16 == 0 and w <= 3840 and h <= 2160


def test_strip_code_fences():
    assert common.strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert common.strip_code_fences('  {"a": 1} ') == '{"a": 1}'


def test_clean_latex():
    tex = r"""
\section{Method} % our approach
We use a \textbf{transformer}~\cite{vaswani2017} with loss $\mathcal{L} = \sum_i \ell_i$.
\begin{figure}[t]\centering\includegraphics{x.pdf}\caption{drop me}\end{figure}
See Section~\ref{sec:exp}.
"""
    out = common.clean_latex(tex)
    assert "Method" in out and "transformer" in out and r"$\mathcal{L}" in out
    for gone in ("our approach", r"\cite", "drop me", r"\ref", r"\textbf"):
        assert gone not in out


def test_parse_pages():
    assert common.parse_pages("3-5,8", 10) == [2, 3, 4, 7]
    assert common.parse_pages("9-12", 10) == [8, 9]
    assert common.parse_pages(None, 3) == [0, 1, 2]


def test_venues():
    icml = common.load_venue("icml")
    assert icml["columns"] == 2
    assert common.venue_figure_width(icml, "single") == 3.25
    assert common.venue_figure_width(icml, "double") == 6.75
    assert common.load_venue("NeurIPS")["key"] == "neurips"
    with pytest.raises(ValueError):
        common.load_venue("nope")


def test_load_methodology(tmp_path):
    txt = tmp_path / "m.txt"; txt.write_text("  Our method ...  ")
    assert common.load_methodology(None, str(txt)) == "Our method ..."
    tex = tmp_path / "m.tex"; tex.write_text(r"\section{Method} Text \cite{x}.")
    assert common.load_methodology(None, str(tex)) == "Method\n Text ."
    assert common.load_methodology("inline wins", str(txt)) == "inline wins"
    with pytest.raises(FileNotFoundError):
        common.load_methodology(None, str(tmp_path / "missing.txt"))
    with pytest.raises(ValueError):
        common.load_methodology(None, None)


def test_load_methodology_pdf(tmp_path):
    pytest.importorskip("pypdf")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdf = tmp_path / "paper.pdf"
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        fig = plt.figure(figsize=(4, 2))
        fig.text(0.1, 0.5, "Our encoder uses attention", fontsize=12)
        fig.savefig(pdf)
        plt.close(fig)
    text = common.load_methodology(None, str(pdf), pages="1")
    assert "encoder" in text and "attention" in text


def test_usage_summary():
    providers.reset_usage()
    providers._record("m", 10, 5)
    providers._record("m", 1, 1, images=1)
    (row,) = providers.usage_summary()
    assert row == {"model": "m", "calls": 2, "input_tokens": 11, "output_tokens": 6, "images": 1}
    providers.reset_usage()
