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
from cl.runtime.log.exceptions.user_error import UserError
from cl.runtime.testing.regression_guard import RegressionGuard
from cl.convince.entries.entry import Entry
from cl.convince.entries.entry_key import EntryKey
from cl.convince.entries.entry_type_key import EntryTypeKey


def test_get_entry_id():
    """Test EntryKey.create_key method."""

    guard = RegressionGuard()

    # Record type
    record_type = "SampleEntryType"

    # Check with type and title only
    guard.write(EntryKey.get_entry_id(record_type, "Sample Title"))

    # Check with body
    guard.write(EntryKey.get_entry_id(record_type, "Sample Title", body="Sample Body"))

    # Check with data
    guard.write(EntryKey.get_entry_id(record_type, "Sample Title", data="Sample Data"))

    # Check with both
    guard.write(EntryKey.get_entry_id(record_type, "Sample Title", body="Sample Body", data="Sample Data"))

    # Verify
    guard.verify_all()


def test_check_entry_id():
    """Test EntryKey.create_key method."""

    EntryKey.check_entry_id("SampleEntryType: Sample Title")
    EntryKey.check_entry_id("SampleEntryType: Sample Title (MD5: 00000000000000000000000000000000)")
    with pytest.raises(UserError):
        EntryKey.check_entry_id("Sample Title")
    with pytest.raises(UserError):
        EntryKey.check_entry_id("SampleEntryType: Sample Title (MD5: 00000000000000000000000000000000")
    with pytest.raises(UserError):
        EntryKey.check_entry_id("SampleEntryType: Sample Title (MD5: 000000000000000000000000000000000")
    with pytest.raises(UserError):
        EntryKey.check_entry_id("SampleEntryType: Sample Title (md5: 0000000000000000000000000000000")


if __name__ == "__main__":
    pytest.main([__file__])
