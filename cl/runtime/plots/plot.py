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

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from cl.runtime.plots.plot_key import PlotKey
from cl.runtime.plots.plot_style_key import PlotStyleKey
from cl.runtime.records.record_mixin import RecordMixin


@dataclass(slots=True, kw_only=True)
class Plot(PlotKey, RecordMixin[PlotKey], ABC):
    """Base class for plot objects."""

    style: PlotStyleKey | None = None
    """Color and layout options."""

    def get_key(self) -> PlotKey:
        return PlotKey(plot_id=self.plot_id)

    @abstractmethod
    def get_view(self) -> None:
        """Return a view object for the plot."""

    @abstractmethod
    def save_png(self) -> None:
        """Save in png format to 'base_dir/plot_id.png'."""
