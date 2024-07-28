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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import openai
from dotenv import load_dotenv


@dataclass(slots=True, init=False)
class Settings:
    """Default settings may be modified before the settings object is passed to the model."""

    openai_api_key: str = field(default=None)
    """API key for OpenAI models."""

    def __init__(self):
        """Load settings from the environment variables and .env file (environment variables take precedence)."""

        # Load additional environment variables from .env file during import of this module,
        # Do not override the environment variables
        load_dotenv(override=False)

        # Package: OpenAI
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Set OpenAI key explicitly because it does not automatically load the variables set by load_dotenv()
        # TODO: Pass OpenAI key to each method to allow code with different settings to run in parallel
        openai.api_key = self.openai_api_key

    @staticmethod
    def load() -> None:
        """Syntactic sugar for creating an instance of Settings class to read .env file
        or environment variable and set global settings such as OpenAI key."""

        # Creating the object reads from .env or environment variables
        # and sets global settings such as OpenAI key.
        Settings()
