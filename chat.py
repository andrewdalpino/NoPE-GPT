import random

from os import path
from functools import partial
from argparse import ArgumentParser
from itertools import chain

import torch

from torch.cuda import is_available as cuda_is_available

from model import LightGPT
from data import CHATML_TEMPLATE, RESPONSE_HEADER
from memory import ChatMemory

DEFAULT_SYSTEM_MESSAGE = (
    "You're a helpful AI assistant named LightGPT. "
    "Your job is to chat and answer questions as accurately as possible. "
)


def main():
    parser = ArgumentParser(description="Chat with the instruction-tuned model.")

    parser.add_argument(
        "--base_checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument(
        "--lora_checkpoint_path", default="./checkpoints/instruct.pt", type=str
    )
    parser.add_argument("--max_tokens", default=500, type=int)
    parser.add_argument("--context_length", default=1024, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top_k", default=500, type=int)
    parser.add_argument("--top_p", default=0.9, type=float)
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

    system_message_tokens = tokenizer.encode(system_message, allowed_special="all")

    response_header_tokens = tokenizer.encode(RESPONSE_HEADER, allowed_special="all")

    newline_token = tokenizer.encode_single_token("\n")

    max_message_history_length = args.context_length - len(system_message_tokens)

    memory = ChatMemory(max_message_history_length)

    generate = partial(
        model.generate,
        max_tokens=args.max_tokens,
        context_length=args.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    while True:
        instruction = input("Enter a prompt: ")

        instruction = CHATML_TEMPLATE.format(role="user", message=instruction)

        instruction_tokens = tokenizer.encode(instruction, allowed_special="all")

        prompt = chain(
            system_message_tokens,
            memory.get_history(),
            instruction_tokens,
            response_header_tokens,
        )

        prompt = torch.tensor(list(prompt), dtype=torch.int64, device=args.device)

        memory.add_message(instruction_tokens)

        message_tokens = response_header_tokens

        for token in generate(prompt):
            message_tokens.append(token)

            if token in stop_tokens:
                break

            out = tokenizer.decode_single_token_bytes(token).decode(
                "utf-8", errors="replace"
            )

            print(out, end="", flush=True)

        message_tokens.append(newline_token)

        print("\n")

        if "y" not in input("Go again? (yes|no): ").lower():
            break

        memory.add_message(message_tokens)


if __name__ == "__main__":
    main()
