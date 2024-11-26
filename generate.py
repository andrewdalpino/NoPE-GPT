import random

from os import path
from argparse import ArgumentParser

import torch

from torch.amp import autocast
from torch.cuda import is_available as cuda_is_available, is_bf16_supported

from model import GPT, GPTWithLoRA
from data import Alpaca

import tiktoken


def main():
    parser = ArgumentParser(
        description="Generate text from the model given a prompt.",
    )

    parser.add_argument("--checkpoint_path", default="./out/checkpoint.pt", type=str)
    parser.add_argument("--lora_path", default=None, type=str)
    parser.add_argument("--max_tokens", default=500, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    torch.set_float32_matmul_precision("high")

    dtype = (
        torch.bfloat16
        if "cuda" in args.device and is_bf16_supported()
        else torch.float32
    )

    forward_context = autocast(device_type=args.device, dtype=dtype)

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    tokenizer = tiktoken.get_encoding(Alpaca.ENCODING)

    checkpoint = torch.load(
        args.checkpoint_path, map_location=args.device, weights_only=True
    )

    model = GPT(**checkpoint["model_args"])

    model = torch.compile(model)

    model.load_state_dict(checkpoint["model"])

    print("Model checkpoint loaded")

    if args.lora_path:
        checkpoint = torch.load(
            args.lora_path, map_location=args.device, weights_only=True
        )

        model = GPTWithLoRA(model, **checkpoint["lora_args"])

        model = torch.compile(model)

        model.load_state_dict(checkpoint["lora"], strict=False)

        print("LoRA checkpoint loaded")

    model.to(args.device)

    model.eval()

    prompt = input("Enter a prompt: ")

    if args.lora_path:
        prompt = Alpaca.PROMPT_TEMPLATE.format(instruction=prompt)

    prompt = tokenizer.encode_ordinary(prompt)

    prompt = torch.tensor(prompt, dtype=torch.int64, device=args.device)

    with forward_context:
        for token in model.generate(
            prompt, args.max_tokens, args.temperature, args.top_k
        ):

            out = tokenizer.decode([token])

            print(out, end="")

    print("\n")


if __name__ == "__main__":
    main()
