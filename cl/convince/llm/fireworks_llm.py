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

import fireworks.client  # noqa
from cl.convince.llm.llm import Llm
from cl.convince.settings.fireworks_settings import FireworksSettings
from dataclasses import dataclass

fireworks.client.api_key = FireworksSettings.instance().api_key
"""Fireworks API key."""


@dataclass(slots=True, kw_only=True)
class FireworksLlm(Llm):
    """Implements Fireworks LLM API."""

    model_name: str | None = None
    """Model name in Fireworks format including version if any, defaults to 'llm_id'."""

    def completion(self, query: str) -> str:

        model_name = self.model_name if self.model_name is not None else self.llm_id
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
        response = fireworks.client.Completion.create(model=f"accounts/fireworks/models/{model_name}", prompt=prompt)
        result = response.choices[0].text
        return result