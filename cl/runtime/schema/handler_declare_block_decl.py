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

import inspect
from dataclasses import dataclass
from typing import Iterable
from typing import List
from inflection import humanize
from inflection import titleize
from memoization import cached
from cl.runtime.records.dataclasses_extensions import missing
from cl.runtime.schema.handler_declare_decl import HandlerDeclareDecl
from cl.runtime.schema.handler_variable_decl import HandlerVariableDecl


@dataclass(slots=True, kw_only=True)
class HandlerDeclareBlockDecl:
    """Handler declaration block in type declaration."""

    handlers: List[HandlerDeclareDecl] = missing()
    """Handler declaration data."""

    @classmethod
    @cached
    def get_type_methods(cls, record_type: type, inherit: bool = False) -> "HandlerDeclareBlockDecl":
        """Extract class public methods."""

        type_members: Iterable[str] = []
        if inherit:
            type_members = dir(record_type)
        else:
            type_members = vars(record_type)

        # Search for methods in type members
        handlers: List[HandlerDeclareDecl] = list()
        for member_name in type_members:
            if member_name.startswith("_") or member_name.startswith("__"):  # Skip all private methods
                continue

            member = getattr(record_type, member_name)
            if inspect.isfunction(member) or inspect.ismethod(member):
                handler = HandlerDeclareDecl()
                handler.name = member_name
                handler.comment = member.__doc__
                handler.static = (
                    # The handler is considered static if it is declared as staticmethod or classmethod.
                    # It does not matter if it is a classmethod or staticmethod from UI perspective
                    # (they are both considered static)
                    isinstance(inspect.getattr_static(record_type, member_name), staticmethod)
                    or isinstance(inspect.getattr_static(record_type, member_name), classmethod)
                )

                # TODO: Add labels support
                handler.label = titleize(humanize(member_name))

                # TODO: Implement for handlers and contents

                if member_name.startswith("run_"):
                    # If method name starts with "run_" consider it a handler
                    handler.type_ = "Job"
                    # Strip the `run_` prefix
                    handler.label = handler.label[4:]
                elif member_name.startswith("view_"):
                    # If method name starts with "view_" consider it a viewer
                    handler.type_ = "Viewer"
                    # Strip the `view_` prefix
                    handler.label = handler.label[5:]
                else:
                    continue

                handlers.append(handler)

                # Process method's return type
                # TODO: Add support of return comment
                if (return_type := member.__annotations__.get("return", None)) is not None:
                    handler.return_ = HandlerVariableDecl.create(value_type=return_type, record_type=record_type)

        return cls(handlers=handlers)
