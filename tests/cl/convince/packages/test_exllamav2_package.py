import os
import time

import pytest
import torch.cuda
from dotenv import load_dotenv

if torch.cuda.is_available():
    from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
    from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_smoke():
    load_dotenv(override=False)
    model_directory = os.path.join(os.getenv("CONFIRMS_MODEL_DIR"), "Llama-2-7b-Chat-GPTQ")

    config = ExLlamaV2Config()
    config.model_dir = model_directory
    config.prepare()

    model = ExLlamaV2(config)
    print("Loading model: " + model_directory)
    model.load()

    tokenizer = ExLlamaV2Tokenizer(config)
    cache = ExLlamaV2Cache(model)
    generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.85
    settings.top_k = 50
    settings.top_p = 0.8
    settings.token_repetition_penalty = 1.15
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

    prompt = "Our story begins in the Scottish town of Auchtermuchty, where once"

    max_new_tokens = 150

    generator.warmup()
    time_begin = time.time()

    output = generator.generate_simple(prompt, settings, max_new_tokens, seed=1234)

    time_end = time.time()
    time_total = time_end - time_begin

    print(output)
    print()
    print(
        f"Response generated in {time_total:.2f} seconds, {max_new_tokens} tokens, {max_new_tokens / time_total:.2f} "
        "tokens/second"
    )


if __name__ == '__main__':
    pytest.main([__file__])
