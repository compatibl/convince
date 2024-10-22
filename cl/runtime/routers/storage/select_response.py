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
from typing import Any
from typing import Dict
from typing import List
from pydantic import BaseModel
from pydantic import Field
from cl.runtime.context.context import Context
from cl.runtime.primitive.case_util import CaseUtil
from cl.runtime.records.class_info import ClassInfo
from cl.runtime.routers.schema.type_request import TypeRequest
from cl.runtime.routers.schema.type_response_util import TypeResponseUtil
from cl.runtime.routers.storage.select_request import SelectRequest
from cl.runtime.serialization.ui_dict_serializer import UiDictSerializer

SelectResponseSchema = Dict[str, Any]
SelectResponseData = List[Dict[str, Any]]


class SelectResponse(BaseModel):
    """Response data type for the /storage/select route."""

    schema_: SelectResponseSchema = Field(..., alias="schema")
    """Schema field of the response data type for the /storage/select route."""

    data: SelectResponseData
    """Data field of the response data type for the /storage/select route."""

    @classmethod
    def get_records(cls, request: SelectRequest) -> SelectResponse:
        """Implements /storage/select route."""

        # Default response when running locally without authorization
        type_decl_dict = TypeResponseUtil.get_type(TypeRequest(name=request.type_, module=request.module, user="root"))

        record_module = request.module
        record_class = request.type_
        record_type = ClassInfo.get_class_type(f"{record_module}.{record_class}")

        # Get database from the current context
        db = Context.current().db

        # TODO (Roman): replace temporary load_all to load_filter
        if not hasattr(db, "load_all"):
            raise RuntimeError(
                f"Currently database need to implement load_all() method for select records by type. "
                f"Database {db.__class__.__name__} doesn't have load_all()."
            )

        # load records by type
        records = db.load_all(record_type)
        records = list(records)

        # TODO: Refactor the code below

        ui_serializer = UiDictSerializer()

        # TODO (Roman): check if we are calling /select somewhere other than the main grid.
        serialized_records = tuple(ui_serializer.serialize_record_for_table(record) for record in records)

        return SelectResponse(schema=type_decl_dict, data=serialized_records).dict(by_alias=True)
