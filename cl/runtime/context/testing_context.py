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
from cl.runtime.backend.core.user_key import UserKey
from cl.runtime.context.context import Context
from cl.runtime.context.env_util import EnvUtil
from cl.runtime.db.dataset_util import DatasetUtil
from cl.runtime.records.class_info import ClassInfo
from cl.runtime.settings.context_settings import ContextSettings
from cl.runtime.settings.settings import is_inside_test


@dataclass(slots=True, kw_only=True)
class TestingContext(Context):
    """
    Utilities for both pytest and unittest.

    Notes:
        - The name TestingContext was selected to avoid Test prefix and does not indicate it is for a specific package
        - This module not itself import pytest or unittest package
    """

    db_class: str | None = None
    """Override for the database class in module.ClassName format."""

    def __post_init__(self):
        """Configure fields that were not specified in constructor."""

        # Do not execute this code on deserialized context instances (e.g. when they are passed to a task queue)
        if not self.is_deserialized:
            # Confirm we are inside a test, error otherwise
            if not is_inside_test:
                raise RuntimeError(f"TestingContext created outside a test.")

            # Get test name in 'module.test_function' or 'module.TestClass.test_method' format inside a test
            context_settings = ContextSettings.instance()

            # For the test, env name is dot-delimited test module, class in snake_case (if any), and method or function
            env_name = EnvUtil.get_env_name()

            # Use test name in dot-delimited format for context_id unless specified by the caller
            if self.context_id is None:
                self.context_id = env_name

            # Set user to env name for unit testing
            self.user = UserKey(username=env_name)

            # TODO: Set log field here explicitly instead of relying on implicit detection of test environment
            log_type = ClassInfo.get_class_type(context_settings.log_class)
            self.log = log_type(log_id=self.context_id)

            # Use database class from settings unless this class provides an override
            if self.db_class is not None:
                db_class = self.db_class
            else:
                db_class = context_settings.db_class

            # Use 'temp' followed by context_id converted to semicolon-delimited format for db_id
            db_id = "temp;" + self.context_id.replace(".", ";")

            # Instantiate a new database object for every test
            db_type = ClassInfo.get_class_type(db_class)
            self.db = db_type(db_id=db_id)

            # Root dataset
            self.dataset = DatasetUtil.root()

    def __enter__(self):
        """Supports 'with' operator for resource disposal."""

        # Call '__enter__' method of base first
        Context.__enter__(self)

        # Do not execute this code on deserialized context instances (e.g. when they are passed to a task queue)
        if not self.is_deserialized:
            # Delete all existing data in temp database and drop DB in case it was not cleaned up
            # due to abnormal termination of the previous test run
            self.db.delete_all_and_drop_db()  # noqa

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Supports 'with' operator for resource disposal."""

        # Do not execute this code on deserialized context instances (e.g. when they are passed to a task queue)
        if not self.is_deserialized:
            # Delete all data in temp database and drop DB to clean up
            self.db.delete_all_and_drop_db()  # noqa

        # Call '__exit__' method of base last
        return Context.__exit__(self, exc_type, exc_val, exc_tb)
