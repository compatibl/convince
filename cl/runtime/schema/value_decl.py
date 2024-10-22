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
from typing import Literal
from typing import Type
from typing_extensions import Self
from cl.runtime.primitive.primitive_util import PrimitiveUtil
from cl.runtime.records.dataclasses_extensions import missing

PrimitiveTypeLiteral = Literal[
    "String", "Double", "Bool", "Int", "Long", "Date", "Time", "DateTime", "UUID", "Binary", "Dict"
]


@dataclass(slots=True, kw_only=True)
class ValueDecl:
    """Value or atomic element declaration."""

    type_: PrimitiveTypeLiteral = missing()
    """Primitive type name."""

    @classmethod
    def create(cls, record_type: Type | str) -> Self:
        """Create an instance based on specified type."""

        if not PrimitiveUtil.is_primitive(record_type):
            raise RuntimeError(f"Primitive field type {record_type} is not supported.")
        return ValueDecl(type_=PrimitiveUtil.get_runtime_name(record_type))  # noqa
