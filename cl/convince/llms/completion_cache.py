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

import collections
import csv
import os
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Iterable
from cl.runtime.records.dataclasses_extensions import field
from cl.runtime.settings.settings import Settings
from cl.runtime.testing.stack_util import StackUtil

_supported_extensions = ["csv"]
"""The list of supported output file extensions (formats)."""

_csv_headers = ["RequestID", "Query", "Completion"]
"""CSV column headers."""


def _error_extension_not_supported(ext: str) -> Any:
    raise RuntimeError(
        f"Extension {ext} is not supported by CompletionCache. "
        f"Supported extensions: {', '.join(_supported_extensions)}"
    )


@dataclass(slots=True, kw_only=True)
class CompletionCache:
    """
    Cache LLM completions for reducing AI cost (disable when testing the LLM itself)

    Notes:
        - After each model call, input and output are recorded in 'channel.completions.csv'
        - The channel may be based on llm_id or include some of all of the LLM settings or their hash
        - If exactly the same input is subsequently found in the completions file, it is used without calling the LLM
        - To record a new completions file, delete the existing one
    """

    channel: str | None = None
    """Dot-delimited string or an iterable of dot-delimited tokens to uniquely identify the cache."""

    ext: str | None = None
    """Output file extension (format) without the dot prefix, defaults to 'csv'."""

    output_path: str | None = None
    """Path for the cache file where completions are stored."""

    __completion_dict: Dict[str, str] = field(default_factory=lambda: {})  # TODO: Set using ContextVars
    """Dictionary of completions indexed by query."""

    def __post_init__(self):
        """
        Load the completions file from disk once on construction. New completions added to this instance
        are written to disk but not reused.
        """

        # Find base_path=dir_path/test_module by examining call stack for test function signature test_*
        base_dir = StackUtil.get_base_dir(allow_missing=True)

        # If not found, use base path relative to project root
        if base_dir is None:
            project_root = Settings.get_project_root()
            base_dir = os.path.join(project_root, "completions")

        if self.ext is not None:
            # Remove dot prefix if specified
            self.ext = self.ext.removeprefix(".")
            if self.ext not in _supported_extensions:
                _error_extension_not_supported(self.ext)
        else:
            # Use csv if not specified
            self.ext = "csv"

        # Cache file path
        if self.channel is None or self.channel == "":
            cache_filename = f"completions.{self.ext}"
        else:
            cache_filename = f"{self.channel}.completions.{self.ext}"
        self.output_path = os.path.join(base_dir, cache_filename)

        # Load cache file from disk
        self.load_cache_file()

    def add(self, request_id: str, query: str, completion: str, *, trial_id: str | int | None = None) -> None:
        """Add to file even if already exits, the latest will take precedence during lookup."""

        # Check if the file already exists
        is_new = not os.path.exists(self.output_path)

        # If file does not exist, create directory if directory does not exist
        if is_new:
            output_dir = os.path.dirname(self.output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        if self.ext == "csv":
            with open(self.output_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(
                    file,
                    delimiter=",",
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL,
                    escapechar="\\",
                    lineterminator=os.linesep,
                )

                if is_new:
                    # Write the headers if the file is new
                    writer.writerow(self._normalize_eol_for_file(_csv_headers))

                # NOT ADDING THE VALUE TO COMPLETION DICT HERE IS NOT A BUG
                # Because we are not adding to the dict here but only writing to a file,
                # the model will not reuse cached completions within the same session,
                # preventing incorrect measurement of stability

                # Get cache key with trial_id, EOL normalization, and stripped leading and trailing whitespace
                cache_key = self.normalize_key(query, trial_id=trial_id)

                # Normalize EOL character in completion and strip whitespace
                cached_value = self.normalize_eol(completion)

                # Write the new completion without checking if one already exists
                writer.writerow(self._normalize_eol_for_file([request_id, cache_key, cached_value]))

                # Flush immediately to ensure all of the output is on disk in the event of exception
                file.flush()
        else:
            # Should not be reached here because of a previous check in __init__
            _error_extension_not_supported(self.ext)

    def get(self, query: str, *, trial_id: str | int | None = None) -> str | None:
        """Return completion for the specified query if found and None otherwise."""

        # Get cache key with trial_id, EOL normalization, and stripped leading and trailing whitespace
        cache_key = self.normalize_key(query, trial_id=trial_id)

        # Look up with trial ID
        result = self.__completion_dict.get(cache_key, None)

        if result is not None:
            # Normalize EOL character in result and strip leading and trailing whitespace
            result = self.normalize_eol(result)
        return result

    def load_cache_file(self) -> None:
        """Load cache file."""
        if os.path.exists(self.output_path):
            # Populate the dictionary from file if exists but not yet loaded
            with open(self.output_path, mode="r", newline="", encoding="utf-8") as file:
                reader = csv.reader(file, delimiter=",", quotechar='"', escapechar="\\", lineterminator=os.linesep)

                # Read and validate the headers
                headers_in_file = next(reader, None)
                if headers_in_file != _csv_headers:
                    max_len = 20
                    headers_in_file = [h if len(h) < max_len else f"{h[:max_len]}..." for h in headers_in_file]
                    headers_in_file_str = ", ".join(headers_in_file)
                    expected_headers_str = ", ".join(_csv_headers)
                    raise ValueError(
                        f"Expected column headers in completions cache are {expected_headers_str}. "
                        f"Actual headers: {headers_in_file_str}."
                    )

                # Read cached completions, ignoring request_id at position 0
                self.__completion_dict.update(
                    {row[1]: row[2] for row_ in reader if (row := self._normalize_eol_from_file(row_))}
                )

    @classmethod
    def normalize_key(cls, query: str, trial_id: str | int | None = None) -> str:
        """Add trial_id and normalize EOL character and whitespace."""

        # Add trial_id to the beginning of cached query key
        if trial_id is not None:
            result = f"TrialID: {str(trial_id)} {query}"
        else:
            result = query

        # Normalize EOL character and strip leading and trailing whitespace
        result = cls.normalize_eol(result)
        return result

    @classmethod
    def normalize_eol(cls, data: Iterable[str] | str | None):
        """Unify line endings in data to \n."""
        if data is None:
            return None
        if not isinstance(data, str) and isinstance(data, collections.abc.Iterable):
            # If data is iterable return list of adjusted elements
            return [cls.normalize_eol(x) for x in data]
        else:
            # Replace endings format to \n
            data = data.replace("\r\r\n", "\n")
            data = data.replace("\r\n", "\n")
            return data

    @classmethod
    def _normalize_eol_for_file(cls, data: Iterable[str] | str | None):
        """Replace \n in data to os.linesep."""
        if data is None:
            return None
        if not isinstance(data, str) and isinstance(data, collections.abc.Iterable):
            # If data is iterable return list of adjusted elements
            return [cls._normalize_eol_for_file(x) for x in data]
        else:
            # Raise an exception if data contains os.linesep characters that are not \n, since
            # they will be lost after normalization.
            if os.linesep != "\n" and os.linesep in data:
                raise RuntimeError("Can not normalize data contains os.linesep characters that are not \\n.")

            # Replace \n to os.linesep
            adjusted_data = data.replace("\n", os.linesep)

            return adjusted_data

    @classmethod
    def _normalize_eol_from_file(cls, data: Iterable[str] | str | None):
        """Replace \n in data to os.linesep."""
        if data is None:
            return None
        if not isinstance(data, str) and isinstance(data, collections.abc.Iterable):
            # If data is iterable return list of adjusted elements
            return [cls._normalize_eol_from_file(x) for x in data]
        else:
            # Replace os.linesep in data to \n
            adjusted_data = data.replace(os.linesep, "\n")

            return adjusted_data
