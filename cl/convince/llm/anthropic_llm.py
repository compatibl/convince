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

from anthropic import Anthropic
from cl.convince.llm.llm import Llm
from cl.convince.settings.anthropic_settings import AnthropicSettings
from dataclasses import dataclass

_client = Anthropic(api_key=AnthropicSettings.instance().api_key)

api_key = AnthropicSettings.instance().api_key
"""Anthropic API key."""


@dataclass(slots=True, kw_only=True)
class AnthropicLlm(Llm):
    """Implements Anthropic LLM API."""

    model_name: str | None = None
    """Model name in Anthropic format including version if any, defaults to 'llm_id'."""

    max_tokens: int = 1024
    """Maximum number of tokens the model will generate in response to the query."""

    def completion(self, query: str) -> str:

        model_name = self.model_name if self.model_name is not None else self.llm_id
        messages = [{"role": "user", "content": query}]
        response = _client.messages.create(
            model=model_name,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        if len(response.content) != 1:
            raise RuntimeError(f"More than one response message received for query: {query}: {str(response)}")
        result = response.content[0].text
        return result