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
from logging import Logger
from typing import ClassVar
from typing import Protocol
from cl.runtime.db.protocols import DbProtocol


class ProgressProtocol(Protocol):
    """Protocol implemented by objects used for progress reporting."""


class ContextProtocol(Protocol):
    """Protocol implemented by context objects providing logging, database, dataset, and progress reporting."""

    current: ClassVar[ContextProtocol]
    """Return current context, error message if not set."""

    logger: Logger
    """Return the logger provided by the context."""

    db: DbProtocol | None
    """Return the default database of the context or None if not set."""

    dataset: str
    """Return the default dataset of the context or None if not set."""

    progress: ProgressProtocol | None
    """Return the progress reporting interface of the context or None if not set."""
