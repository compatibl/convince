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

from pydantic import BaseModel
from cl.runtime.primitive.case_util import CaseUtil
from cl.runtime.records.dataclasses_extensions import missing


class RunErrorResponseItem(BaseModel):
    """Data type for a single item in the response list for the /tasks/run route in case of an error."""

    task_run_id: str | None = missing()
    """Task run id."""

    key: str | None = missing()
    """Key of the record."""

    name: str | None = missing()
    """Name of the exception."""

    status_code: int | None = missing()
    """Status code of the task."""

    message: str | None = missing()
    """Message of the exception."""

    stack_trace: str | None = missing()
    """Stack trace of the exception."""

    class Config:
        alias_generator = CaseUtil.snake_to_pascal_case
        populate_by_name = True
