import random

from functools import partial
from argparse import ArgumentParser

import torch

from torch.cuda import is_available as cuda_is_available

from colored import fore_rgb, style

from src.nope_gpt.model import NoPEGPT
from memory import BufferWindowMemory

DEFAULT_SYSTEM_MESSAGE = (
    "You're a helpful AI assistant named NoPEGPT. "
    "Your job is to chat and answer questions as accurately as possible. "
    "If you don't know the answer, say 'I don't know'. "
)

RESPONSE_HEADER = "<|im_start|>assistant\n"

WHITE = (255, 255, 255)


def main():
    parser = ArgumentParser(description="Chat with the instruction-tuned model.")

    parser.add_argument(
        "--base_checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument(
        "--lora_checkpoint_path", default="./checkpoints/instruct.pt", type=str
    )
    parser.add_argument("--max_tokens", default=2000, type=int)
    parser.add_argument("--colorize_tokens", action="store_true")
    parser.add_argument("--context_length", default=2048, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top_k", default=500, type=int)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--repeat_penalty", default=0.1, type=float)
    parser.add_argument("--repeat_window", default=50, type=int)
    parser.add_argument("--max_message_history", default=4, type=int)
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
        args.base_checkpoint_path, map_location=args.device, weights_only=False
    )

    tokenizer = checkpoint["tokenizer"]

    model = NoPEGPT(**checkpoint["model_args"])

    model.resize_token_embeddings(tokenizer.vocabulary_size)

    model.add_lora_parameters(**checkpoint["lora_args"])

    model.load_state_dict(checkpoint["model"])

    model.merge_lora_parameters()

    model.to(args.device)

    print("Model checkpoint loaded")

    model.eval()

    system_message = input("Enter a system message: ")

    if not system_message:
        system_message = DEFAULT_SYSTEM_MESSAGE

    system_message = {
        "role": "system",
        "content": system_message,
    }

    memory = BufferWindowMemory(args.max_message_history)

    generate = partial(
        model.generate,
        max_tokens=args.max_tokens,
        context_length=args.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        repeat_window=args.repeat_window,
    )

    while True:
        instruction = input("Enter a prompt: ")

        instruction_message = {
            "role": "user",
            "content": instruction,
        }

        messages = [system_message] + memory.get_history() + [instruction_message]

        prompt = torch.tensor(list(prompt), dtype=torch.int64, device=args.device)

        for token, probability in generate(prompt):
            token, probability = token.item(), probability.item()

            if token == tokenizer.tokenizer.eot_token_id:
                break

            if args.colorize_tokens:
                intensity = int(probability * 255)

                r, g, b = 255 - intensity, 0, intensity
            else:
                r, g, b = WHITE

            color = fore_rgb(r, g, b)

            out = tokenizer.decode_single_token_bytes(token).decode(
                "utf-8", errors="replace"
            )

            print(f"{color}{out}{style("reset")}", end="", flush=True)

        print("\n")

        if "y" not in input("Go again? (yes|no): ").lower():
            break

        memory.add_message(response_message)


if __name__ == "__main__":
    main()
