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
from typing_extensions import Self
from cl.runtime.records.dataclasses_extensions import field
from cl.runtime.records.dataclasses_extensions import missing
from cl.runtime.schema.enum_decl_key import EnumDeclKey
from cl.runtime.schema.field_decl import FieldDecl
from cl.runtime.schema.member_decl import MemberDecl
from cl.runtime.schema.module_decl_key import ModuleDeclKey
from cl.runtime.schema.type_decl_key import TypeDeclKey
from cl.runtime.schema.value_decl import ValueDecl


@dataclass(slots=True, kw_only=True)
class ElementDecl(MemberDecl):  # TODO: Consider renaming to TypeFieldDecl or FieldDecl
    """Type element declaration."""

    name: str = missing()
    """Element name."""

    label: str | None = missing()
    """Element label. If not specified, name is used instead."""

    comment: str | None = missing()
    """Element comment. Contains addition information."""

    vector: bool | None = missing()  # TODO: Replace by container field with enum values vector/array, dict, DF
    """Flag indicating variable size array (vector) container."""

    optional: bool | None = missing()
    """Flag indicating optional element."""

    optional_vector_element: bool | None = missing()  # TODO: Rename to optional_element or optional_field
    """Flag indicating optional vector item element."""

    additive: bool | None = missing()
    """Optional flag indicating if the element is additive and that the total column can be shown in the UI."""

    format_: str | None = field(name="Format")  # TODO: Use Python interpolated string format
    """Specifies UI Format for the element."""

    alternate_of: str | None = missing()
    """Link current element to AlternateOf element. In the editor these elements will be treated as a choice."""

    @classmethod
    def create(cls, field_decl: FieldDecl) -> Self:
        """Create ElementDecl from FieldDecl."""

        result = ElementDecl()
        result.name = field_decl.name
        result.label = field_decl.label
        result.comment = field_decl.comment
        result.optional = field_decl.optional_field
        result.optional_vector_element = field_decl.optional_values
        result.additive = None  # TODO: Support in metadata
        result.format_ = field_decl.formatter
        result.alternate_of = None  # TODO: Support in metadata

        if field_decl.field_kind == "primitive":
            # Primitive type
            result.value = ValueDecl.create(field_decl.field_type)
        else:
            # Complex type
            module_name, type_name = field_decl.field_type.rsplit(".", 1)

            if field_decl.field_kind == "enum":
                module_key = ModuleDeclKey(module_name=module_name)
                result.enum = EnumDeclKey(module=module_key, name=type_name)
            elif field_decl.field_kind == "key":
                module_key = ModuleDeclKey(module_name=module_name)
                result.key_ = TypeDeclKey(module=module_key, name=type_name)
            elif field_decl.field_kind == "data":
                module_key = ModuleDeclKey(module_name=module_name)
                result.data = TypeDeclKey(module=module_key, name=type_name)
            else:
                raise RuntimeError(f"Unsupported field kind {field_decl.field_kind} for field {field_decl.name}.")

        if field_decl.container_type is None:
            result.vector = False
        elif field_decl.container_type == "list":
            result.vector = True
        elif field_decl.container_type == "dict":
            # TODO (Roman): Use another way to define the dict field. This is currently legacy format.
            result.value = ValueDecl(type_="Dict")
        else:
            raise RuntimeError(f"Unsupported container type {field_decl.container_type} for field {field_decl.name}.")

        return result
