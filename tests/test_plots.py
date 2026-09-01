import warnings

import pytest
from PIL import Image

import plot_generator

CONFIGS = {
    "bar": {"type": "bar", "data": {"BERT": 92.3, "GPT-4": 88.1}, "ylabel": "F1 (%)", "title": "F1"},
    "grouped_bar": {"type": "bar", "data": [[1, 2], [3, 4]], "labels": ["A", "B"], "series_names": ["s1", "s2"]},
    "line": {"type": "line", "series": {"A": {"1": 0.5, "2": 0.7}, "B": [[1, 0.4], [2, 0.6]]}},
    "scatter": {"type": "scatter", "series": {"A": [[1, 2], [2, 3]]}},
    "heatmap": {"type": "confusion_matrix", "data": [[5, 1], [2, 7]], "xlabels": ["a", "b"], "ylabels": ["a", "b"]},
    "box": {"type": "box", "data": [[1, 2, 3, 4, 5], [2, 3, 4, 5, 9]], "labels": ["A", "B"]},
    "violin": {"type": "violin", "data": [[1, 2, 3, 4, 5], [2, 3, 4, 5, 9]], "labels": ["A", "B"]},
    "histogram": {"type": "histogram", "data": {"A": [1, 2, 2, 3], "B": [2, 3, 4]}},
    "radar": {"type": "radar", "data": {"M1": [1, 2, 3]}, "categories": ["x", "y", "z"]},
}


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_plot_types_render_without_warnings(name, tmp_path):
    out = tmp_path / f"{name}.png"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_generator.generate_plot(dict(CONFIGS[name]), str(out))
    assert out.stat().st_size > 0


def test_venue_width_controls_figure_size(tmp_path):
    single, double = tmp_path / "s.png", tmp_path / "d.png"
    plot_generator.generate_plot({**CONFIGS["bar"], "venue": "icml", "width": "single"}, str(single))
    plot_generator.generate_plot({**CONFIGS["bar"], "venue": "icml", "width": "double"}, str(double))
    assert Image.open(single).size[0] < Image.open(double).size[0]
