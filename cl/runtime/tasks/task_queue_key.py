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
from typing import Type
from cl.runtime.records.dataclasses_extensions import missing
from cl.runtime.records.key_mixin import KeyMixin


@dataclass(slots=True, kw_only=True)
class TaskQueueKey(KeyMixin):
    """
    Invokes the 'execute' method of the submitted tasks sequentially or in parallel with other tasks.

    Notes:
        - The task may be invoked in a different process, thread or machine than the submitting code
          and must be able to acquire the resources required by its 'execute' method in all of these cases
        - The queue creates a new TaskRun record every time the task is submitted
        - The TaskRun record is periodically updated by the queue with the run status and result
        - The TaskRun record must never be modified by the task itself
    """

    queue_id: str = missing()
    """Unique task queue identifier."""

    @classmethod
    def get_key_type(cls) -> Type:
        return TaskQueueKey
