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
from typing import Optional

import torch
from huggingface_hub import hf_hub_download

from cl.convince.llm.llm import Llm
from cl.convince.settings import Settings

if torch.cuda.is_available():
    from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
    from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler

    @dataclass
    class LlamaGpuLlm(Llm):
        """LLAMA 2 models loaded using exllama adapter."""

        temperature: float = field(default=None)
        """Model temperature (note that for GPT models zero value does not mean reproducible answers)."""

        seed: int = field(default=None)
        """Model seed (use the same seed to reproduce the answer)."""

        _llm: ExLlamaV2BaseGenerator = field(default=None)

        _settings: Settings = field(default=None)

        def load_model(self):
            """Load model after fields have been set."""

            settings = Settings()

            # Skip if already loaded
            if self._llm is None:
                # Set repo_id and GPU layers based on name
                model_filename = self.model_type
                if model_filename.startswith("llama-2-7b-chat"):
                    repo_id = "TheBloke/Llama-2-7B-Chat-GPTQ"
                elif model_filename.startswith("llama-2-13b-chat"):
                    repo_id = "TheBloke/Llama-2-13B-Chat-GPTQ"
                elif model_filename.startswith("llama-2-70b-chat"):
                    repo_id = "TheBloke/Llama-2-70B-Chat-GPTQ"
                else:
                    raise RuntimeError(f"Repo not specified for model type {self.model_type}")

                model_path = settings.get_model_path(model_filename, check_exists=False)
                if not os.path.exists(model_path):
                    print(
                        f"Model {model_filename} is not found in {model_path} and will be downloaded from Hugging Face."
                        f"This may take from tens of minutes to hours time depending on network speed."
                    )
                    hf_hub_download(repo_id=repo_id, local_dir=settings.model_dir, local_dir_use_symlinks=False)

                config = ExLlamaV2Config()
                config.model_dir = model_path
                config.prepare()

                model = ExLlamaV2(config)
                model.load()

                tokenizer = ExLlamaV2Tokenizer(config)
                cache = ExLlamaV2Cache(model)
                generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

                settings = ExLlamaV2Sampler.Settings()
                settings.temperature = self.temperature if self.temperature is not None else 0.2
                settings.top_k = 70
                settings.top_p = 0.85
                settings.token_repetition_penalty = 1.15
                settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

                generator.warmup()

                self._llm = generator
                self._settings = settings

        def completion(self, question: str, *, prompt: Optional[str] = None) -> str:
            """Simple completion with optional prompt."""

            # Load model (multiple calls do not need to reload)
            self.load_model()

            # TODO: implement prompt
            max_new_tokens = 128
            answer = self._llm.generate_simple(question, self._settings, max_new_tokens, seed=self.seed)

            return answer
