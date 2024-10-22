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

import datetime as dt
from dataclasses import dataclass
from typing import Dict
from typing import List
from cl.runtime.records.dataclasses_extensions import field
from cl.runtime.records.dataclasses_extensions import missing
from stubs.cl.runtime.records.for_dataclasses.stub_dataclass_data import StubDataclassData
from stubs.cl.runtime.records.for_dataclasses.stub_dataclass_derived_record import StubDataclassDerivedRecord
from stubs.cl.runtime.records.for_dataclasses.stub_dataclass_dict_fields import stub_dataclass_data_dict_factory
from stubs.cl.runtime.records.for_dataclasses.stub_dataclass_dict_fields import stub_dataclass_date_dict_factory
from stubs.cl.runtime.records.for_dataclasses.stub_dataclass_dict_fields import (
    stub_dataclass_derived_record_dict_factory,
)
from stubs.cl.runtime.records.for_dataclasses.stub_dataclass_dict_fields import stub_dataclass_key_dict_factory
from stubs.cl.runtime.records.for_dataclasses.stub_dataclass_dict_fields import stub_dataclass_record_dict_factory
from stubs.cl.runtime.records.for_dataclasses.stub_dataclass_record import StubDataclassRecord
from stubs.cl.runtime.records.for_dataclasses.stub_dataclass_record import StubDataclassRecordKey


def stub_dataclass_str_dict_list_factory() -> List[str]:
    """Create stub values."""
    return ["abc", "def"]


def stub_dataclass_float_dict_list_factory() -> List[Dict[str, float]]:
    """Create stub values."""
    return [
        {"a": 1.0},
        {"b": 2.0},
    ]


def stub_dataclass_date_dict_list_factory() -> List[Dict[str, dt.date]]:
    """Create stub values."""
    return [
        stub_dataclass_date_dict_factory(),
        stub_dataclass_date_dict_factory(),
    ]


def stub_dataclass_data_dict_list_factory() -> List[Dict[str, StubDataclassData]]:
    """Create stub values."""
    return [
        stub_dataclass_data_dict_factory(),
        stub_dataclass_data_dict_factory(),
    ]


def stub_dataclass_key_dict_list_factory() -> List[Dict[str, StubDataclassRecordKey]]:
    """Create stub values."""
    return [
        stub_dataclass_key_dict_factory(),
        stub_dataclass_key_dict_factory(),
    ]


def stub_dataclass_record_dict_list_factory() -> List[Dict[str, StubDataclassRecord]]:
    """Create stub values."""
    return [
        stub_dataclass_record_dict_factory(),
        stub_dataclass_record_dict_factory(),
    ]


def stub_dataclass_derived_record_dict_list_factory() -> List[Dict[str, StubDataclassDerivedRecord]]:
    """Create stub values."""
    return [
        stub_dataclass_derived_record_dict_factory(),
        stub_dataclass_derived_record_dict_factory(),
    ]


@dataclass(slots=True, kw_only=True)
class StubDataclassDictListFields(StubDataclassRecord):
    """Stub record whose elements are dictionaries."""

    float_dict_list: List[Dict[str, float]] = field(default_factory=stub_dataclass_float_dict_list_factory)
    """Stub field."""

    date_dict_list: List[Dict[str, dt.date]] = field(default_factory=stub_dataclass_date_dict_list_factory)
    """Stub field."""

    record_dict_list: List[Dict[str, StubDataclassRecord]] = field(
        default_factory=stub_dataclass_record_dict_list_factory
    )
    """Stub field."""

    derived_record_dict_list: List[Dict[str, StubDataclassDerivedRecord]] = field(
        default_factory=stub_dataclass_derived_record_dict_list_factory
    )
    """Stub field."""
