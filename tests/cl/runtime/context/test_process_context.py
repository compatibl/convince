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
from cl.runtime.context.process_context import ProcessContext
from cl.runtime.settings.context_settings import ContextSettings

context_settings = ContextSettings.instance()


def test_process_context():
    """Smoke test."""

    # Check that ProcessContext cannot be invoked inside a test
    with pytest.raises(RuntimeError):
        with ProcessContext():
            pass


if __name__ == "__main__":
    pytest.main([__file__])
