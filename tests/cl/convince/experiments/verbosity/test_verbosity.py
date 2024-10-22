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
from cl.runtime.testing.pytest.pytest_fixtures import local_dir_fixture
from stubs.cl.convince.experiments.stub_llms import get_stub_mini_llms


def _get_question(i: int):
    return f"{i} times {i}"


def _get_simple_prompt(i: int):
    question = f"{i} times {i}"
    return f"What is {_get_question(i)}?"


def _get_extended_prompt(i: int):
    return (
        f"Reply with the answer for ```{_get_question(i)}``. Your reply must include numerical value for the answer "
        f"(not words) with no other comments or other text."
    )


def test_verbosity(local_dir_fixture):
    """Test for verbosity."""

    with TestingContext():

        reps = 2
        plot = GroupBarPlot(plot_id="verbosity")
        plot.values = []
        plot.bar_labels = []
        plot.group_labels = []
        stub_mini_llms = get_stub_mini_llms()
        for llm in stub_mini_llms:

            # Calculate
            simple_sum = sum(llm.completion(_get_simple_prompt(i + 1)) == str(pow(i + 1, 2)) for i in range(reps))
            extended_sum = sum(llm.completion(_get_extended_prompt(i + 1)) == str(pow(i + 1, 2)) for i in range(reps))

            # Add to plot
            plot.bar_labels.extend([llm.llm_id] * 2)
            plot.group_labels.extend(["Simple", "Extended"])
            plot.values.extend([simple_sum / reps, extended_sum / reps])

        # Save
        plot.save_png()


if __name__ == "__main__":
    pytest.main([__file__])
