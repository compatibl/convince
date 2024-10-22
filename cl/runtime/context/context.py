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

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Type
from cl.runtime.backend.core.user_key import UserKey
from cl.runtime.context.context_key import ContextKey
from cl.runtime.db.db_key import DbKey
from cl.runtime.db.protocols import TKey
from cl.runtime.db.protocols import TRecord
from cl.runtime.log.exceptions.user_error import UserError
from cl.runtime.log.log_entry import LogEntry
from cl.runtime.log.log_entry_level_enum import LogEntryLevelEnum
from cl.runtime.log.log_key import LogKey
from cl.runtime.log.user_log_entry import UserLogEntry
from cl.runtime.records.dataclasses_extensions import missing
from cl.runtime.records.protocols import KeyProtocol
from cl.runtime.records.protocols import RecordProtocol
from cl.runtime.records.protocols import is_key
from cl.runtime.records.record_mixin import RecordMixin
from cl.runtime.settings.context_settings import ContextSettings

root_context_types_str = """
The following root context types can be used in the outermost 'with' clause:
    - ProcessContext: Context for launching a process, use in __main__
    - TestingContext: Context for running unit tests
"""

_context_stack: ContextVar[Optional[List["Context"]]] = ContextVar("context_stack", default=None)
"""
Context adds self to the stack on __enter__ and removes self on __exit__.
Each asynchronous context has its own stack.
"""


@contextmanager
def request_cycle_context() -> Iterator[None]:
    """Context manager to create isolated queue of contexts"""
    token = _context_stack.set([])
    yield
    _context_stack.reset(token)


@dataclass(slots=True, kw_only=True)
class Context(ContextKey, RecordMixin[ContextKey]):
    """Protocol implemented by context objects providing logging, database, dataset, and progress reporting."""

    user: UserKey = missing()
    """Current user, 'Context.current().user' is used if not specified."""

    log: LogKey = missing()
    """Log of the context, 'Context.current().log' is used if not specified."""

    db: DbKey = missing()
    """Database of the context, 'Context.current().db' is used if not specified."""

    dataset: str = missing()
    """Dataset of the context, 'Context.current().dataset' is used if not specified."""

    is_deserialized: bool = False
    """Use this flag to determine if this context instance has been deserialized from data."""

    def __post_init__(self):
        """Set fields to their values in 'Context.current()' if not specified."""

        # Do not execute this code on deserialized context instances (e.g. when they are passed to a task queue)
        if not self.is_deserialized:
            # Set fields that are not specified to their values from 'Context.current()'
            if self.user is None:
                self._root_context_field_not_set_error("user")
                self.user = Context.current().user
            if self.log is None:
                self._root_context_field_not_set_error("log")
                self.log = Context.current().log
            if self.db is None:
                self._root_context_field_not_set_error("db")
                self.db = Context.current().db
            if self.dataset is None:
                self._root_context_field_not_set_error("dataset")
                self.dataset = Context.current().dataset

        # Replace fields that are set as keys by records from storage
        # First, load 'db' field of this context using 'Context.current()'
        if is_key(self.db):
            self.db = Context.current().load_one(DbKey, self.db)

        # After this all remaining fields can be loaded using database from this context
        if is_key(self.log):
            self.log = self.load_one(LogKey, self.log)

    def get_key(self) -> ContextKey:
        return ContextKey(context_id=self.context_id)

    @classmethod
    def current(cls):
        """Return the current context or None if not set."""
        context_stack = _context_stack.get()
        if context_stack and len(context_stack) > 0:
            return context_stack[-1]
        else:
            raise RuntimeError(
                "Current context is not set, use 'with' clause with a root context type to set."
                + root_context_types_str
            )

    def __enter__(self):
        """Supports 'with' operator for resource disposal."""

        context_stack = _context_stack.get()
        if context_stack is None:
            # Context activated without middleware, create a new context stack
            context_stack = []
            _context_stack.set(context_stack)

        # Check if self is already the current context
        if context_stack and context_stack[-1] is self:
            raise RuntimeError("The context activated using 'with' operator is already current.")

        # Set current context on entering 'with Context(...)' clause
        context_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Supports 'with' operator for resource disposal."""

        if exc_val is not None:
            # Save log entry to the database
            # Get log entry type and level
            if isinstance(exc_val, UserError):
                log_type = UserLogEntry
                level = LogEntryLevelEnum.USER_ERROR
            else:
                log_type = LogEntry
                level = LogEntryLevelEnum.ERROR

            # Create log entry
            log_entry = log_type(  # noqa
                message=str(exc_val),
                level=level,
            )
            log_entry.init()

            # Save occurred error to self db
            self.save_one(log_entry)

        context_stack = _context_stack.get()

        if context_stack is None or not bool(context_stack):
            raise RuntimeError("Current context must not be cleared inside 'with Context(...)' clause.")

        # Restore the previous current context on exiting from 'with Context(...)' clause
        current_context = context_stack.pop()
        if current_context is not self:
            raise RuntimeError("Current context must only be modified by 'with Context(...)' clause.")

        # TODO: Support resource disposal for the database
        if self.db is not None:
            # TODO: Finalize approach to disposal self.db.disconnect()
            pass

        # Return False to propagate exception to the caller
        return False

    def get_logger(self, name: str) -> logging.Logger:
        """Get logger for the specified name, invoke with __name__ as the argument."""
        return self.log.get_logger(name)  # noqa

    def load_one(
        self,
        record_type: Type[TRecord],
        record_or_key: TRecord | KeyProtocol | None,
        *,
        dataset: str | None = None,
        identity: str | None = None,
        is_key_optional: bool = False,
        is_record_optional: bool = False,
    ) -> TRecord | None:
        """
        Load a single record using a key (if a record is passed instead of a key, it is returned without DB lookup)

        Args:
            record_type: Record type to load, error if the result is not this type or its subclass
            record_or_key: Record (returned without lookup) or key in object, tuple or string format
            dataset: If specified, append to the root dataset of the database
            identity: Identity token for database access and row-level security
            is_key_optional: If True, return None when key is none found instead of an error
            is_record_optional: If True, return None when record is not found instead of an error
        """
        return self.db.load_one(  # noqa
            record_type,
            record_or_key,
            dataset=dataset,
            identity=identity,
            is_key_optional=is_key_optional,
            is_record_optional=is_record_optional,
        )

    def load_many(
        self,
        record_type: Type[TRecord],
        records_or_keys: Iterable[TRecord | KeyProtocol | tuple | str | None] | None,
        *,
        dataset: str | None = None,
        identity: str | None = None,
    ) -> Iterable[TRecord | None] | None:
        """
        Load records using a list of keys (if a record is passed instead of a key, it is returned without DB lookup),
        the result must have the same order as 'records_or_keys'.

        Args:
            record_type: Record type to load, error if the result is not this type or its subclass
            records_or_keys: Records (returned without lookup) or keys in object, tuple or string format
            dataset: If specified, append to the root dataset of the database
            identity: Identity token for database access and row-level security
        """
        return self.db.load_many(  # noqa
            record_type,
            records_or_keys,
            dataset=dataset,
            identity=identity,
        )

    def load_all(
        self,
        record_type: Type[TRecord],
        *,
        dataset: str | None = None,
        identity: str | None = None,
    ) -> Iterable[TRecord | None] | None:
        """
        Load all records of the specified type and its subtypes (excludes other types in the same DB table).

        Args:
            record_type: Type of the records to load
            dataset: If specified, append to the root dataset of the database
            identity: Identity token for database access and row-level security
        """
        return self.db.load_all(  # noqa
            record_type,
            dataset=dataset,
            identity=identity,
        )

    def load_filter(
        self,
        record_type: Type[TRecord],
        filter_obj: TRecord,
        *,
        dataset: str | None = None,
        identity: str | None = None,
    ) -> Iterable[TRecord]:
        """
        Load records where values of those fields that are set in the filter match the filter.

        Args:
            record_type: Record type to load, error if the result is not this type or its subclass
            filter_obj: Instance of 'record_type' whose fields are used for the query
            dataset: If specified, append to the root dataset of the database
            identity: Identity token for database access and row-level security
        """
        return self.db.load_filter(  # noqa
            record_type,
            filter_obj,
            dataset=dataset,
            identity=identity,
        )

    def save_one(
        self,
        record: RecordProtocol | None,
        *,
        dataset: str | None = None,
        identity: str | None = None,
    ) -> None:
        """
        Save records to storage.

        Args:
            record: Record or None.
            dataset: Target dataset as a delimited string, list of levels, or None
            identity: Identity token for database access and row-level security
        """
        self.db.save_one(  # noqa
            record,
            dataset=dataset,
            identity=identity,
        )

    def save_many(
        self,
        records: Iterable[RecordProtocol],
        *,
        dataset: str | None = None,
        identity: str | None = None,
    ) -> None:
        """
        Save records to storage.

        Args:
            records: Iterable of records.
            dataset: Target dataset as a delimited string, list of levels, or None
            identity: Identity token for database access and row-level security
        """
        self.db.save_many(  # noqa
            records,
            dataset=dataset,
            identity=identity,
        )

    def delete_one(
        self,
        key_type: Type[TKey],
        key: TKey | KeyProtocol | tuple | str | None,
        *,
        dataset: str | None = None,
        identity: str | None = None,
    ) -> None:
        """
        Delete one record for the specified key type using its key in one of several possible formats.

        Args:
            key_type: Key type to delete, used to determine the database table
            key: Key in object, tuple or string format
            dataset: If specified, append to the root dataset of the database
            identity: Identity token for database access and row-level security
        """
        self.db.delete_one(  # noqa
            key_type,
            key,
            dataset=dataset,
            identity=identity,
        )

    def delete_many(
        self,
        keys: Iterable[KeyProtocol] | None,
        *,
        dataset: str | None = None,
        identity: str | None = None,
    ) -> None:
        """
        Delete records using an iterable of keys.

        Args:
            keys: Iterable of keys.
            dataset: Target dataset as a delimited string, list of levels, or None
            identity: Identity token for database access and row-level security
        """
        self.db.delete_many(  # noqa
            keys,
            dataset=dataset,
            identity=identity,
        )

    def delete_all_and_drop_db(self) -> None:
        """
        IMPORTANT: !!! DESTRUCTIVE - THIS WILL PERMANENTLY DELETE ALL RECORDS WITHOUT THE POSSIBILITY OF RECOVERY

        Notes:
            This method will not run unless both db_id and database start with 'temp_db_prefix'
            specified using Dynaconf and stored in 'DbSettings' class
        """
        # Additional check in context in case a custom database implementation does not check it
        self.error_if_not_temp_db(self.db.db_id)
        self.db.delete_all_and_drop_db()  # noqa

    def _root_context_field_not_set_error(self, field_name: str) -> None:
        """Error message about a Context field not set."""
        if type(self) is not Context:
            raise RuntimeError(
                f"""
Field '{field_name}' of the context class '{type(self).__name__}' is not set.
The context in the outermost 'with' clause (root context) must set all fields
of the Context class. Inside the 'with' clause, these fields will be populated
from the current context.
"""
                + root_context_types_str
            )

    @classmethod
    def error_if_not_temp_db(cls, db_id_or_database_name: str) -> None:
        """Confirm that database id or database name matches temp_db_prefix, error otherwise."""
        context_settings = ContextSettings.instance()
        temp_db_prefix = context_settings.db_temp_prefix
        if not db_id_or_database_name.startswith(temp_db_prefix):
            raise RuntimeError(
                f"Destructive action on database not permitted because db_id or database name "
                f"'{db_id_or_database_name}' does not match temp_db_prefix '{temp_db_prefix}' "
                f"specified in Dynaconf database settings ('DbSettings' class)."
            )
