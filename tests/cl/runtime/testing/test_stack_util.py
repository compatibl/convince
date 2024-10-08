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
from cl.runtime.testing.stack_util import StackUtil

# TODO: Add tests for get_base_path


def test_stack_util():
    """Method name matches module name, shortened path"""
    dir_name = os.path.dirname(__file__)
    expected_result = os.path.join(dir_name, "test_stack_util")
    result = StackUtil.get_base_dir()
    assert result == expected_result


def test_get_base_path_in_function():
    """Function name does not match module name, two-token path."""
    dir_name = os.path.dirname(__file__)
    expected_result = os.path.join(dir_name, "test_stack_util", "test_get_base_path_in_function")
    result = StackUtil.get_base_dir()
    assert result == expected_result


class TestClass:
    """Stub pytest class."""

    def test_stack_util(self):
        """Method name matches module name, still three-token path as they are not next to each other."""
        """Function name does not match module name, two-token path."""
        dir_name = os.path.dirname(__file__)
        expected_result = os.path.join(dir_name, "test_stack_util", "test_class", "test_stack_util")
        result = StackUtil.get_base_dir()
        assert result == expected_result

    def test_get_base_path_in_method(self):
        """Method name does not match class name or module name, three-token path"""
        dir_name = os.path.dirname(__file__)
        expected_result = os.path.join(dir_name, "test_stack_util", "test_class", "test_get_base_path_in_method")
        result = StackUtil.get_base_dir()
        assert result == expected_result


class TestStackUtil:
    """Stub pytest class with name matching the module."""

    def test_stack_util(self):
        """All three match, one-token path."""
        dir_name = os.path.dirname(__file__)
        expected_result = os.path.join(dir_name, "test_stack_util")
        result = StackUtil.get_base_dir()
        assert result == expected_result

    def test_get_base_path_in_method(self):
        """Method name does not match class name or module name which match, two-token path"""
        dir_name = os.path.dirname(__file__)
        expected_result = os.path.join(dir_name, "test_stack_util", "test_get_base_path_in_method")
        result = StackUtil.get_base_dir()
        assert result == expected_result


if __name__ == "__main__":
    pytest.main([__file__])
