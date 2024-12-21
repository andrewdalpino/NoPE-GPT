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

    parser.add_argument("--checkpoint_path", default="./out/checkpoint.pt", type=str)
    parser.add_argument("--lora_path", default=None, type=str)
    parser.add_argument("--max_tokens", default=200, type=int)
    parser.add_argument("--num_candidates", default=3, type=int)
    parser.add_argument("--beam_width", default=16, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    torch.set_float32_matmul_precision("high")

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

        model.merge_lora_parameters()

        print("LoRA checkpoint loaded")

    model.to(args.device)

    model.eval()

    while True:
        prompt = input("Enter a prompt: ")

        if args.lora_path:
            prompt = Alpaca.PROMPT_TEMPLATE.format(instruction=prompt)

        prompt = tokenizer.encode_ordinary(prompt)

        prompt = torch.tensor(prompt, dtype=torch.int64, device=args.device)

        candidates = model.beam_search(
            prompt,
            args.max_tokens,
            args.num_candidates,
            args.beam_width,
        )

        for i, candidate in enumerate(candidates, start=1):
            print(f"Sequence #{i} ({candidate.probability:.4}):")

            out = tokenizer.decode(candidate.tokens.tolist()).strip()

            print(out, end="\n\n")

        print("\n")

        if "y" not in input("Try again? (yes|no): ").lower():
            break


if __name__ == "__main__":
    main()
