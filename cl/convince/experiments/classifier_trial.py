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

from dataclasses import dataclass
from cl.convince.experiments.experiment_key import ExperimentKey
from cl.convince.experiments.trial import Trial
from cl.runtime.records.dataclasses_extensions import missing
from cl.convince.experiments.trial_key import TrialKey


@dataclass(slots=True, kw_only=True)
class ClassifierTrial(Trial):
    """Records the result of a single category (class label) assignment trial."""

    category: bool = missing()
    """Category (class label) assigned by this trial."""