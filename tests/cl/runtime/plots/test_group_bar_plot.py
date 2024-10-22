# Copyright (C) 2023-present The Project Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from cl.runtime.context.testing_context import TestingContext
from cl.runtime.plots.group_bar_plot import GroupBarPlot
from cl.runtime.plots.group_bar_plot_style import GroupBarPlotStyle
from cl.runtime.testing.pytest.pytest_fixtures import local_dir_fixture


def test_single_group(local_dir_fixture):
    with TestingContext() as context:
        group_bar_plot = GroupBarPlot(plot_id="group_bar_plot")
        group_bar_plot.group_labels = ["Single Group"] * 2
        group_bar_plot.bar_labels = ["Bar 1", "Bar 2"]
        group_bar_plot.values = [85.5, 92]
        group_bar_plot.save_png()


def test_dark_theme(local_dir_fixture):
    with TestingContext() as context:
        group_bar_plot = GroupBarPlot(plot_id="group_bar_plot")
        group_bar_plot.group_labels = ["Single Group"] * 2
        group_bar_plot.bar_labels = ["Bar 1", "Bar 2"]
        group_bar_plot.values = [85.5, 92]
        group_bar_plot.style = GroupBarPlotStyle(dark_theme=True)
        group_bar_plot.save_png()


def test_4_groups_2_bars(local_dir_fixture):
    with TestingContext() as context:
        num_groups = 4
        num_bars = 2

        bar_labels = []

        for i in range(num_bars):
            bar_labels += [f"Metric {i + 1}"] * num_groups

        group_labels = [f"Model {i + 1}" for i in range(num_groups)] * num_bars

        group_bar_plot = GroupBarPlot(plot_id="group_bar_plot")
        group_bar_plot.title = "Model Comparison"
        group_bar_plot.bar_labels = bar_labels
        group_bar_plot.group_labels = group_labels
        group_bar_plot.values = [
            10,
            20,
            20,
            40,  # "Metric 1"
            20,
            30,
            25,
            30,  # "Metric 2"
        ]
        group_bar_plot.save_png()


def test_4_groups_5_bars(local_dir_fixture):
    with TestingContext() as context:
        num_groups = 4
        num_bars = 5

        bar_labels = []

        for i in range(num_bars):
            bar_labels += [f"Metric {i + 1}"] * num_groups

        group_labels = [f"Model {i + 1}" for i in range(num_groups)] * num_bars

        group_bar_plot = GroupBarPlot(plot_id="group_bar_plot")
        group_bar_plot.title = "Model Comparison"
        group_bar_plot.bar_axis_label = "Metrics"
        group_bar_plot.value_axis_label = "Models"
        group_bar_plot.bar_labels = bar_labels
        group_bar_plot.group_labels = group_labels
        group_bar_plot.values = [
            85.5,
            92,
            70,
            83.7,  # "Metric 1"
            89,
            95.3,
            77,
            95,  # "Metric 2"
            81,
            93.6,
            75,
            63.5,  # "Metric 3"
            85.5,
            98.8,
            78,
            83.7,  # "Metric 4"
            79.5,
            90,
            72.4,
            81.8,  # "Metric 5"
        ]
        group_bar_plot.value_ticks = list(range(0, 101, 10))
        group_bar_plot.style = GroupBarPlotStyle(dark_theme=True)
        group_bar_plot.save_png()


if __name__ == "__main__":
    pytest.main([__file__])
