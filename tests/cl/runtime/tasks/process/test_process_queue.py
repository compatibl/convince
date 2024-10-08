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
from cl.runtime.tasks.process.process_queue import ProcessQueue
from cl.runtime.tasks.static_method_task import StaticMethodTask
from stubs.cl.runtime.decorators.stub_handlers import StubHandlers


def test_smoke():
    """Smoke test."""

    with TestingContext() as context:
        obj = StubHandlers(stub_id="abc")
        context.save_one(obj)

        method_callable = StubHandlers.static_handler_1a
        task = StaticMethodTask.create(task_id="abc", record_type=StubHandlers, method_callable=method_callable)

        queue = ProcessQueue()
        task_run_key = queue.submit_task(task)


if __name__ == "__main__":
    pytest.main([__file__])
