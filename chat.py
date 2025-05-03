import random

from functools import partial
from argparse import ArgumentParser
from itertools import chain
import copy

import torch

from torch.cuda import is_available as cuda_is_available

from colored import fore_rgb, style

from model import LightGPT
from data import CHATML_TEMPLATE, RESPONSE_HEADER
from memory import ShortTermMemory

DEFAULT_SYSTEM_MESSAGE = (
    "You're a helpful AI assistant named LightGPT. "
    "Your job is to chat and answer questions as accurately as possible. "
)

WHITE = (255, 255, 255)


def main():
    parser = ArgumentParser(description="Chat with the instruction-tuned model.")

    parser.add_argument(
        "--base_checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument(
        "--lora_checkpoint_path", default="./checkpoints/instruct.pt", type=str
    )
    parser.add_argument("--max_tokens", default=500, type=int)
    parser.add_argument("--colorize_tokens", action="store_true")
    parser.add_argument("--context_length", default=1024, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top_k", default=500, type=int)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--repeat_penalty", default=0.1, type=float)
    parser.add_argument("--repeat_window", default=50, type=int)
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

    model = LightGPT(**checkpoint["model_args"])

    model = torch.compile(model)

    model.load_state_dict(checkpoint["model"])

    print("Model checkpoint loaded")

    checkpoint = torch.load(
        args.lora_checkpoint_path, map_location=args.device, weights_only=False
    )

    tokenizer = checkpoint["tokenizer"]

    im_end_token = tokenizer.encode_single_token("<|im_end|>")
    eot_token = tokenizer.eot_token

    stop_tokens = frozenset({im_end_token, eot_token})

    model.resize_token_embeddings(tokenizer.n_vocab)
    model.token_embeddings.load_state_dict(checkpoint["token_embeddings"])

    model.add_lora_parameters(**checkpoint["lora_args"])
    model.load_state_dict(checkpoint["lora"], strict=False)
    model.merge_lora_parameters()

    print("LoRA checkpoint loaded")

    model.to(args.device)

    model.eval()

    system_message = input("Enter a system message: ")

    if not system_message:
        system_message = DEFAULT_SYSTEM_MESSAGE

    system_message = CHATML_TEMPLATE.format(role="system", message=system_message)

    system_message = tokenizer.encode(system_message, allowed_special="all")

    response_header = tokenizer.encode(RESPONSE_HEADER, allowed_special="all")

    newline_token = tokenizer.encode_single_token("\n")

    max_token_history = args.context_length - len(system_message)

    memory = ShortTermMemory(max_tokens=max_token_history)

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
        print(f"Short-term memory utilization: {memory.utilization:.2%}")

        instruction = input("Enter a prompt: ")

        instruction_message = CHATML_TEMPLATE.format(role="user", message=instruction)

        instruction_message = tokenizer.encode(
            instruction_message, allowed_special="all"
        )

        prompt = chain(
            system_message,
            memory.get_history(),
            instruction_message,
            response_header,
        )

        prompt = torch.tensor(list(prompt), dtype=torch.int64, device=args.device)

        memory.add_message(instruction_message)

        response_message = copy.copy(response_header)

        for token, probability in generate(prompt):
            token, probability = token.item(), probability.item()

            response_message.append(token)

            if token in stop_tokens:
                break

            if args.colorize_tokens:
                intensity = int(probability * 255)

                r, g, b = 255 - intensity, 0, intensity
            else:
                r, g, b = WHITE

            color = fore_rgb(r, g, b)

            print(f"{color}{out}{style("reset")}", end="", flush=True)

            out = tokenizer.decode_single_token_bytes(token).decode(
                "utf-8", errors="replace"
            )

            print(out, end="", flush=True)

        response_message.append(newline_token)

        print("\n")

        if "y" not in input("Go again? (yes|no): ").lower():
            break

        memory.add_message(response_message)


if __name__ == "__main__":
    main()
