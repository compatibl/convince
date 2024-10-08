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

import re
from enum import Enum
from enum import IntEnum
from typing import Any
from typing import Dict
from typing import Final
from cl.runtime.records.protocols import is_key


class StringValueCustomType(IntEnum):
    """Custom types supported for string representation."""

    data = 0
    """Data type."""

    dict = 1
    """Dict type."""

    list = 2
    """Vector type."""

    date = 3
    """Date type."""

    datetime = 4
    """Datetime type."""

    time = 5
    """Time type."""

    bool = 6
    """Bool type."""

    uuid = 7
    """UUID type."""

    enum = 8
    """Enum type."""

    bytes = 9
    """Binary type."""

    int = 10
    """Integer type."""

    float = 11
    """Float type."""

    key = 12
    """Key type."""


CUSTOM_TYPE_VALUE_TO_NAME: Final[Dict[StringValueCustomType, str]] = {StringValueCustomType.dict: "json"}
"""Enum value to name mapping."""

CUSTOM_TYPE_NAME_TO_VALUE: Final[Dict[str, StringValueCustomType]] = {
    v: k for k, v in CUSTOM_TYPE_VALUE_TO_NAME.items()
}
"""Name to enum value mapping. Reversed mapping."""


class StringValueParser:
    """Parser for string value representations of custom types."""

    @classmethod
    def add_type_prefix(cls, value: str, type_: StringValueCustomType | None) -> str:
        """Add type prefix to value that is a string representation of object of type type_."""

        if type_ is None:
            return value

        # Check type name in alias mapping
        type_name = (
            type_name_alias if ((type_name_alias := CUSTOM_TYPE_VALUE_TO_NAME.get(type_)) is not None) else type_.name
        )

        type_prefix = f"```{type_name} "
        return type_prefix + value

    @classmethod
    def parse(cls, value: str) -> (str, StringValueCustomType | None):
        """
        Check if value is a string representation of some custom type and parse it to separated objects:
            value without type and value type.

        Examples:
            "```bool True" -> "True", bool
            "True" -> "True", None
            "any_string_without_prefix" -> "any_string_without_prefix", None
        """

        # check if value starts with type info prefix using regex
        typed_value_pattern = re.compile("```(?P<type>.*?) .*")
        typed_value_match = typed_value_pattern.match(value)

        if typed_value_match:
            # get custom type name according to pattern
            value_custom_type = typed_value_match.group("type")

            # remove type prefix from value
            value_without_prefix = value.removeprefix(f"```{value_custom_type} ")

            # Check custom type in alias mapping
            value_custom_type = (
                custom_type
                if ((custom_type := CUSTOM_TYPE_NAME_TO_VALUE.get(value_custom_type)) is not None)
                else StringValueCustomType[value_custom_type]
            )

            return value_without_prefix, value_custom_type
        else:
            # return unmodified value and custom type None
            return value, None

    @classmethod
    def get_custom_type(cls, value: Any) -> StringValueCustomType | None:
        """Determine custom_type of value."""
        if value.__class__.__name__ == "date":
            return StringValueCustomType.date
        elif value.__class__.__name__ == "datetime":
            return StringValueCustomType.datetime
        elif value.__class__.__name__ == "time":
            return StringValueCustomType.time
        elif value.__class__.__name__ == "bool":
            return StringValueCustomType.bool
        elif value.__class__.__name__ == "int":
            return StringValueCustomType.int
        elif value.__class__.__name__ == "float":
            return StringValueCustomType.float
        elif value.__class__.__name__ == "UUID":
            return StringValueCustomType.uuid
        elif value.__class__.__name__ == "bytes":
            return StringValueCustomType.bytes
        elif is_key(value):
            return StringValueCustomType.key
        elif hasattr(value, "__slots__"):
            return StringValueCustomType.data
        elif isinstance(value, dict):
            return StringValueCustomType.dict
        elif isinstance(value, Enum):
            return StringValueCustomType.enum
        elif hasattr(value, "__iter__"):
            return StringValueCustomType.list
