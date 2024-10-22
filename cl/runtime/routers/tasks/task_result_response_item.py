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

from __future__ import annotations
from typing import Any
from typing import Iterable
from typing import List
from typing import cast
from pydantic import BaseModel
from cl.runtime import Context
from cl.runtime.primitive.case_util import CaseUtil
from cl.runtime.routers.tasks.task_result_request import TaskResultRequest
from cl.runtime.tasks.task import Task
from cl.runtime.tasks.task_run import TaskRun
from cl.runtime.tasks.task_run_key import TaskRunKey


class TaskResultResponseItem(BaseModel):
    """Data type for a single item in the response list for the /tasks/run/result route."""

    # TODO: Decide on permitted result formats
    result: Any
    """Result of the task run."""

    task_run_id: str
    """Task run id."""

    key: str | None
    """Key of the record."""

    class Config:
        alias_generator = CaseUtil.snake_to_pascal_case
        populate_by_name = True

    @classmethod
    def get_task_results(cls, request: TaskResultRequest) -> List[TaskResultResponseItem]:
        """Get results for tasks in request."""

        # Get current context
        context = Context.current()

        task_run_keys = [TaskRunKey(task_run_id=x) for x in request.task_run_ids]  # TODO: Use =UUID(x)
        task_runs = cast(Iterable[TaskRun], context.load_many(TaskRun, task_run_keys))

        response_items = []
        for task_run in task_runs:
            task_obj = context.load_one(Task, task_run.task)

            response_items.append(
                TaskResultResponseItem(
                    result=task_run.result,
                    task_run_id=str(task_run.task_run_id),
                    key=task_obj.key_str if hasattr(task_obj, "key_str") else None,
                ),
            )

        return response_items
