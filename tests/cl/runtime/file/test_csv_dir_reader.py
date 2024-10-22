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
import os
from cl.runtime.context.env_util import EnvUtil
from cl.runtime.context.testing_context import TestingContext
from cl.runtime.file.csv_dir_reader import CsvDirReader
from cl.runtime.settings.settings import Settings
from stubs.cl.runtime import StubDataclassDerivedRecord
from stubs.cl.runtime import StubDataclassRecord
from stubs.cl.runtime import StubDataclassRecordKey


def test_csv_dir_reader():
    """Test CsvDirReader class."""

    # Create a new instance of local cache for the test
    with TestingContext() as context:
        env_dir = EnvUtil.get_env_dir()
        dir_reader = CsvDirReader(dir_path=env_dir)
        dir_reader.read()

        # Verify
        # TODO: Check count using load_all or count method of Db when created
        for i in range(1, 3):
            key = StubDataclassRecordKey(id=f"base_id_{i}")
            record = context.load_one(StubDataclassRecord, key)
            assert record == StubDataclassRecord(id=f"base_id_{i}")
        for i in range(1, 3):
            key = StubDataclassRecordKey(id=f"derived_id_{i}")
            record = context.load_one(StubDataclassRecord, key)
            assert record == StubDataclassDerivedRecord(
                id=f"derived_id_{i}", derived_field=f"test_derived_field_value_{i}"
            )


if __name__ == "__main__":
    pytest.main([__file__])
