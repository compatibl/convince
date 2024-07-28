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
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


def test_smoke():
    """Confirm that fireworks integration package is installed correctly."""

    model = "gpt-4o-mini"
    temperature = 0  # Do not use 0 except for a smoke test

    llm_request = [
        {"role": "system", "content": "Act as a multiplication table and return the result only"},
        {"role": "user", "content": "2 times 2"}
    ]

    client = OpenAI()
    llm_response = client.chat.completions.create(
        model=model,
        messages=llm_request,
        temperature=temperature
    )
    llm_response_message = llm_response.choices[0].message.content
    assert "4" in llm_response_message


if __name__ == '__main__':
    pytest.main([__file__])
