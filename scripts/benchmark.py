"""Benchmark Steerling generation speed using AutoModel."""

import time

import torch
from transformers import AutoModel, AutoTokenizer

from steerling.inference.causal_diffusion import SteerlingGenerator

model_id ="asalam91/steerling-test"
seed = 42
gen_length = 128
steps = 128
temperature = 0.0

print(f"Loading model from {model_id}...")
model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
generator = SteerlingGenerator.from_model(model, tokenizer, device="cuda")
print(generator)

prompts = [
    "The key to understanding artificial intelligence is",
    "The theory of relativity states that",
    "In machine learning, a transformer model works by",
    "Climate change is caused primarily by",
    "To build a successful startup, you need to",
]

print(f"\n{'='*70}")
print(f"BENCHMARK  |  seed={seed}  gen_length={gen_length}  steps={steps}")
print(f"{'='*70}\n")

# Warmup
torch.manual_seed(seed)
_ = generator.generate("warmup", gen_length=gen_length, steps=steps, temperature=temperature)
torch.cuda.synchronize()

total_tokens = 0
total_time = 0.0

for i, prompt in enumerate(prompts):
    torch.manual_seed(seed)
    prompt_len = len(tokenizer.encode(prompt))

    torch.cuda.synchronize()
    t0 = time.time()
    output = generator.generate(prompt, gen_length=gen_length, steps=steps, temperature=temperature)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    text = generator.decode(output, prompt_len=prompt_len)
    n_tokens = len(tokenizer.encode(text))
    tps = n_tokens / elapsed if elapsed > 0 else 0
    total_tokens += n_tokens
    total_time += elapsed

    print(f"[{i}] {elapsed:.2f}s | {n_tokens} tok | {tps:.1f} tok/s")
    print(f"    Prompt: {prompt}")
    print(f"    Output: {text[:150]}")
    print()

print(f"{'='*70}")
print(f"TOTAL: {total_tokens} tokens in {total_time:.2f}s ({total_tokens / total_time:.1f} tok/s)")
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak")
print(f"{'='*70}")
