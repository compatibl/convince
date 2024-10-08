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
from typing import ClassVar
from openai import OpenAI
from cl.convince.llms.llm import Llm
from cl.convince.settings.openai_settings import OpenaiSettings


@dataclass(slots=True, kw_only=True)
class GptLlm(Llm):
    """Implements GPT LLM API."""

    model_name: str | None = None
    """Model name in OpenAI format including version if any, defaults to 'llm_id'."""

    _client: ClassVar[OpenAI] = None
    """OpenAI client instance."""

    def uncached_completion(self, request_id: str, query: str) -> str:
        """Perform completion without CompletionCache lookup, call completion instead."""

        # Prefix a unique RequestID to the model for audit log purposes and
        # to stop model provider from caching the results
        query_with_request_id = f"RequestID: {request_id}\n\n{query}"

        model_name = self.model_name if self.model_name is not None else self.llm_id
        messages = [{"role": "user", "content": query_with_request_id}]

        client = self._get_client()
        response = client.chat.completions.create(model=model_name, messages=messages)

        result = response.choices[0].message.content
        return result

    @classmethod
    def _get_client(cls) -> OpenAI:
        """Instantiate and cache the OpenAI client instance."""
        if cls._client is None:
            cls._client = OpenAI(
                api_key=OpenaiSettings.instance().api_key,
            )
        return cls._client
