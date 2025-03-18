import random

from os import path
from argparse import ArgumentParser

import torch

from torch.cuda import is_available as cuda_is_available

from model import LightGPT
from data import SmolTalk


def main():
    parser = ArgumentParser(
        description="Chat with the instruction-tuned model.",
    )

    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--lora_path", default="./checkpoints/instruct.pt", type=str)
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
        args.checkpoint_path, map_location=args.device, weights_only=False
    )

    model = LightGPT(**checkpoint["model_args"])

    state_dict = checkpoint["model"]

    # Compensate for poorly designed PyTorch compiled state dicts.
    for key in list(state_dict.keys()):
        state_dict[key.replace("_orig_mod.", "")] = state_dict.pop(key)

    model.load_state_dict(state_dict)

    print("Model checkpoint loaded")

    checkpoint = torch.load(
        args.lora_path, map_location=args.device, weights_only=False
    )

    tokenizer = checkpoint["tokenizer"]

    eos_indices = {tokenizer.eot_token}

    model = (
        model.resize_token_embeddings(tokenizer.n_vocab)
        .add_lora_parameters(**checkpoint["lora_args"])
    )

    model.token_embeddings.load_state_dict(checkpoint["token_embeddings"])
    model.load_state_dict(checkpoint["lora"], strict=False)

    model = model.merge_lora_parameters()

    print("LoRA checkpoint loaded")

    model.to(args.device)

    model.eval()

    system_message = input("Enter a system message: ")

    if not system_message:
        system_message = "You're a helpful AI assistant. Your job is to assist the user with their queries."

    system_message = SmolTalk.PROMPT_TEMPLATE.format(
        role="system", message=system_message
    )

    while True:
        instruction = input("Enter a prompt: ")

        instruction = SmolTalk.PROMPT_TEMPLATE.format(role="user", message=instruction)

        prompt = system_message + instruction

        prompt = tokenizer.encode(prompt, allowed_special="all")

        prompt = torch.tensor(prompt, dtype=torch.int64, device=args.device)

        generator = model.generate(
            prompt,
            args.max_tokens,
            args.context_length,
            args.temperature,
            args.top_k,
            args.top_p,
            eos_indices,
        )

        for token in generator:
            out = tokenizer.decode_single_token_bytes(token).decode(
                "utf-8", errors="replace"
            )

            print(out, end="", flush=True)

            if token.item() == tokenizer.n_vocab - 1:
                generator.close()

        print("\n")

        if "y" not in input("Go again? (yes|no): ").lower():
            break


if __name__ == "__main__":
    main()
