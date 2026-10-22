import asyncio
import contextlib

from transformers import AutoTokenizer
import torch

from gl_cde import (
    SamplingParams,
    init_engine
)

chat_history = [
    {"role": "system", "content": "You are a helpful AI chat assistant."},
    {"role": "user", "content": "Hi how are you today?"},
]
tokenizer = AutoTokenizer.from_pretrained("guidelabs/steerling-8b-instruct", trust_remote_code=True)
prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
tokens = tokenizer.tokenize(prompt)
ids = tokenizer.convert_tokens_to_ids(tokens)
sp = SamplingParams(max_tokens=256)
llm = init_engine()


async def main() -> None:
    print("Starting generation with the following prompt", prompt)
    tokens = []
    with torch.no_grad():
        async with llm:
            async with contextlib.aclosing(llm.generate(ids, sp)) as gen:
                async for output in gen:
                    tokens.extend(output.tokens)
    print("Generated text:", tokenizer.decode(tokens, remove_special_tokens=False))


asyncio.run(main())
