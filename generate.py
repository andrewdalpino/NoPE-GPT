import random

from os import path
from argparse import ArgumentParser

import torch

from torch.cuda import is_available as cuda_is_available

from model import GPT, GPTWithLoRA
from data import Alpaca

import tiktoken


def main():
    parser = ArgumentParser(
        description="Generate text from the model given a prompt.",
    )

    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--lora_path", default=None, type=str)
    parser.add_argument("--max_tokens", default=2000, type=int)
    parser.add_argument("--context_length", default=1024, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top_k", default=500, type=int)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    torch.set_float32_matmul_precision("high")

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    checkpoint = torch.load(
        args.checkpoint_path, map_location=args.device, weights_only=True
    )

    tokenizer = tiktoken.get_encoding(checkpoint["token_encoding"])

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

        model.merge_lora_parameters()

        print("LoRA checkpoint loaded")

    model.to(args.device)

    model.eval()

    while True:
        prompt = input("Enter a prompt: ")

        if args.lora_path:
            context = input("Additional context (leave blank for none): ")

            if len(context) > 0:
                prompt = Alpaca.PROMPT_TEMPLATE_WITH_INPUT.format(
                    input=context, instruction=prompt
                )
            else:
                prompt = Alpaca.PROMPT_TEMPLATE.format(instruction=prompt)

        prompt = tokenizer.encode_ordinary(prompt)

        prompt = torch.tensor(prompt, dtype=torch.int64, device=args.device)

        for token in model.generate(
            prompt,
            args.max_tokens,
            args.context_length,
            args.temperature,
            args.top_k,
            args.top_p,
        ):
            out = tokenizer.decode_single_token_bytes(token).decode(
                "utf-8", errors="replace"
            )

            print(out, end="", flush=True)

        print("\n")

        if "y" not in input("Go again? (yes|no): ").lower():
            break


if __name__ == "__main__":
    main()
