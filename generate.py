import random

from os import path
from argparse import ArgumentParser

import torch

from torch.cuda import is_available as cuda_is_available

from model import LightGPT, LightGPTInstruct
from data import SmolTalk

import tiktoken

from tiktoken import Encoding


def main():
    parser = ArgumentParser(
        description="Generate text from the model given a prompt.",
    )

    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--lora_path", default=None, type=str)
    parser.add_argument("--max_tokens", default=1000, type=int)
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

    eos_indices = {tokenizer.eot_token}

    model = LightGPT(**checkpoint["model_args"])

    # Compensate for poorly designed PyTorch compiled state dicts.
    for key in list(checkpoint["model"].keys()):
        checkpoint["model"][key.replace("_orig_mod.", "")] = checkpoint["model"].pop(
            key
        )

    model.load_state_dict(checkpoint["model"])

    print("Model checkpoint loaded")

    if args.lora_path:
        tokenizer = Encoding(
            name=tokenizer.name,
            pat_str=tokenizer._pat_str,
            mergeable_ranks=tokenizer._mergeable_ranks,
            special_tokens={
                **tokenizer._special_tokens,
                "<|im_start|>": tokenizer.n_vocab,
                "<|im_end|>": tokenizer.n_vocab + 1,
            },
        )

        eos_indices = {*eos_indices, tokenizer.n_vocab + 1}

        checkpoint = torch.load(
            args.lora_path, map_location=args.device, weights_only=True
        )

        model = LightGPTInstruct(model, **checkpoint["lora_args"])

        model.load_state_dict(checkpoint["lora"], strict=False)

        model.merge_lora_parameters()

        print("LoRA checkpoint loaded")

    model.to(args.device)

    model.eval()

    while True:
        prompt = input("Enter a prompt: ")

        if args.lora_path:
            instruction = SmolTalk.PROMPT_TEMPLATE.format(
                role="system",
                message="You're an AI assistant for text re-writing. Rewrite the input text to make it more professional and formal while retaining its essential content.",
            )

            instruction += SmolTalk.PROMPT_TEMPLATE.format(role="user", message=prompt)

            prompt = instruction

        prompt = tokenizer.encode(prompt, allowed_special="all")

        prompt = torch.tensor(prompt, dtype=torch.int64, device=args.device)

        for token in model.generate(
            prompt,
            args.max_tokens,
            args.context_length,
            args.temperature,
            args.top_k,
            args.top_p,
            eos_indices,
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
