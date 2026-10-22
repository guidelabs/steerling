import asyncio
import contextlib

from gl_cde import SamplingParams, init_engine, init_tokenizer


async def main() -> None:
    chat_history = [
        {"role": "system", "content": "You are a helpful AI chat assistant.<|endofchunk|>"},
        {"role": "user", "content": "Hi how are you today?<|endofchunk|>"},
    ]
    tokenizer = init_tokenizer()
    prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    assert isinstance(prompt, str)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    sp = SamplingParams(max_tokens=256)
    llm = init_engine()
    print("Starting generation with the following prompt", prompt)
    tokens = []
    async with llm, contextlib.aclosing(llm.generate(ids, sp)) as gen:
        async for output in gen:
            tokens.extend(output.tokens)
    print("Generated text:", tokenizer.decode(tokens, skip_special_tokens=False))


if __name__ == "__main__":
    asyncio.run(main())
