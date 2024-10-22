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
from fastapi import FastAPI
from starlette.testclient import TestClient
from cl.runtime.context.testing_context import TestingContext
from cl.runtime.primitive.datetime_util import DatetimeUtil
from cl.runtime.routers.tasks import tasks_router
from cl.runtime.routers.tasks.run_response_item import handler_queue
from cl.runtime.routers.tasks.task_status_request import TaskStatusRequest
from cl.runtime.routers.tasks.task_status_response_item import TaskStatusResponseItem
from cl.runtime.serialization.string_serializer import StringSerializer
from cl.runtime.tasks.instance_method_task import InstanceMethodTask
from cl.runtime.tasks.task_run import TaskRun
from cl.runtime.tasks.task_status_enum import TaskStatusEnum
from stubs.cl.runtime import StubHandlers
from stubs.cl.runtime.records.for_dataclasses.stub_dataclass_handlers_key import StubHandlersKey

# Create handler task
task = InstanceMethodTask.create(
    task_id="abc",
    record_or_key=StubHandlersKey(stub_id="abc"),
    method_callable=StubHandlers.run_instance_method_1a,
)

# Get handler task key
key_serializer = StringSerializer()
task_key_str = key_serializer.serialize_key(task.get_key())

# Create task run records
t = DatetimeUtil.now()
queue_key = handler_queue.get_key()
task_runs = [
    TaskRun(queue=queue_key, task=task, submit_time=t, update_time=t, status=TaskStatusEnum.PENDING),
    TaskRun(queue=queue_key, task=task, submit_time=t, update_time=t, status=TaskStatusEnum.FAILED),
    TaskRun(queue=queue_key, task=task, submit_time=t, update_time=t, status=TaskStatusEnum.COMPLETED),
]

# Init task runs
for task_run in task_runs:
    task_run.init()

requests = [
    {
        "task_run_ids": [str(task_run.task_run_id) for task_run in task_runs],
        "db": "DEPRECATED",  # TODO: Review and remove
        "dataset": "",
    }
]


def test_method():
    """Test coroutine for /tasks/run/status route."""

    with TestingContext() as context:
        context.save_one(task)
        context.save_many(task_runs)

        for request in requests:
            request_object = TaskStatusRequest(**request)
            result = TaskStatusResponseItem.get_task_statuses(request_object)

            assert isinstance(result, list)

            for result_item in result:
                assert isinstance(result_item, TaskStatusResponseItem)
                assert result_item.task_run_id is not None
                assert result_item.task_run_id in request_object.task_run_ids
                assert result_item.status_code is not None
                assert result_item.key == task_key_str


def test_api():
    """Test REST API for /tasks/run/status route."""

    # TODO: Use TestingContext instead
    with TestingContext() as context:
        context.save_one(task)
        context.save_many(task_runs)

        test_app = FastAPI()
        test_app.include_router(tasks_router.router, prefix="/tasks", tags=["Tasks"])
        with TestClient(test_app) as test_client:
            for request in requests:
                response = test_client.post("/tasks/run/status", json=request)
                assert response.status_code == 200

                result = response.json()
                assert isinstance(result, list)

                for result_item in result:
                    TaskStatusResponseItem(**result_item)
                    assert isinstance(result_item, dict)
                    assert result_item.get("TaskRunId") is not None
                    assert result_item.get("TaskRunId") in request["task_run_ids"]
                    assert result_item.get("StatusCode") is not None
                    assert result_item.get("Key") == task_key_str


if __name__ == "__main__":
    pytest.main([__file__])
