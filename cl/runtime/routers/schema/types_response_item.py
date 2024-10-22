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

from __future__ import annotations
from typing import List
from inflection import titleize
from pydantic import BaseModel
from cl.runtime.primitive.case_util import CaseUtil
from cl.runtime.routers.user_request import UserRequest
from cl.runtime.schema.schema import Schema


class TypesResponseItem(BaseModel):
    """Single item of the list returned by the /data/types route."""

    name: str
    """Class name (may be customized in settings)."""

    module: str
    """Module path in dot-delimited format (may be customized in settings)."""

    label: str
    """Type label displayed in the UI is humanized class name (may be customized in settings)."""

    class Config:
        alias_generator = CaseUtil.snake_to_pascal_case
        populate_by_name = True

    @classmethod
    def get_types(cls, request: UserRequest) -> List[TypesResponseItem]:
        """Implements /schema/types route."""

        # TODO: Check why UserRequest is not used in this method

        # Get a dictionary of types indexed by short name
        type_dict = Schema.get_type_dict()

        result = [
            TypesResponseItem(
                name=record_type.__name__,
                module=CaseUtil.snake_to_pascal_case(record_type.__module__),
                label=titleize(record_type.__name__),
            )
            for record_type in type_dict.values()
            if hasattr(record_type, "get_key")
        ]
        return result
