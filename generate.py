import random

from functools import partial
from argparse import ArgumentParser

import torch

from torch.cuda import is_available as cuda_is_available

from colored import fore_rgb, style

from model import LightGPT


def main():
    parser = ArgumentParser(
        description="Generate text from the base model by sampling.",
    )

    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--max_tokens", default=1000, type=int)
    parser.add_argument("--colorize_tokens", action="store_true")
    parser.add_argument("--context_length", default=1024, type=int)
    parser.add_argument("--temperature", default=0.9, type=float)
    parser.add_argument("--top_k", default=500, type=int)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--repeat_penalty", default=0.1, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    torch.set_float32_matmul_precision("high")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    checkpoint = torch.load(
        args.checkpoint_path, map_location=args.device, weights_only=False
    )

    tokenizer = checkpoint["tokenizer"]

    model = LightGPT(**checkpoint["model_args"])

    model = torch.compile(model)

    model.load_state_dict(checkpoint["model"])

    print("Model checkpoint loaded")

    model.to(args.device)

    model.eval()

    generate = partial(
        model.generate,
        max_tokens=args.max_tokens,
        context_length=args.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
    )

    while True:
        prompt = input("Enter a prompt: ")

        prompt = tokenizer.encode(prompt, allowed_special="all")

        prompt = torch.tensor(prompt, dtype=torch.int64, device=args.device)

        for token, probability in generate(prompt):
            if token == tokenizer.eot_token:
                break

            out = tokenizer.decode_single_token_bytes(token).decode(
                "utf-8", errors="replace"
            )

            if args.colorize_tokens:
                intensity = int(probability * 255)

                r, g, b = 255 - intensity, 0, intensity
            else:
                r, g, b = 255, 255, 255

            color = fore_rgb(r, g, b)

            print(f"{color}{out}{style("reset")}", end="", flush=True)

        print("\n")

        if "y" not in input("Go again? (yes|no): ").lower():
            break


if __name__ == "__main__":
    main()
