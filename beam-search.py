import random

from functools import partial
from argparse import ArgumentParser

import torch

from torch.cuda import is_available as cuda_is_available

from src.nope_gpt.model import NoPEGPT


def main():
    parser = ArgumentParser(
        description="Generate text from the base model using beam search.",
    )

    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--max_tokens", default=2000, type=int)
    parser.add_argument("--context_length", default=4096, type=int)
    parser.add_argument("--temperature", default=0.9, type=float)
    parser.add_argument("--num_candidates", default=3, type=int)
    parser.add_argument("--beam_width", default=16, type=int)
    parser.add_argument("--length_penalty", default=1.0, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    checkpoint = torch.load(
        args.checkpoint_path, map_location=args.device, weights_only=False
    )

    tokenizer = checkpoint["tokenizer"]

    model = NoPEGPT(**checkpoint["model_args"])

    model = torch.compile(model)

    model.load_state_dict(checkpoint["model"])

    print("Model checkpoint loaded")

    model.to(args.device)

    model.eval()

    beam_search = partial(
        model.beam_search,
        max_tokens=args.max_tokens,
        context_length=args.context_length,
        num_candidates=args.num_candidates,
        beam_width=args.beam_width,
        length_penalty=args.length_penalty,
        eos_indices=tokenizer.stop_tokens,
    )

    while True:
        prompt = input("Enter a prompt: ")

        prompt = tokenizer.tokenize(prompt)

        prompt = torch.tensor(prompt, dtype=torch.int64, device=args.device)

        candidates = beam_search(prompt)

        for i, candidate in enumerate(candidates, start=1):
            print(
                f"Candidate #{i} (Probability: {candidate.cumulative_probability:.4f})"
            )

            print("-" * 40)

            out = tokenizer.decode_tokens(candidate.tokens.tolist())

            print(out)

        print("\n")

        if "y" not in input("Go again? (yes|no): ").lower():
            break


if __name__ == "__main__":
    main()
