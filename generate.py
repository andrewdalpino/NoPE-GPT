import random
from os import path
from argparse import ArgumentParser

import torch

from torch.amp import autocast
from torch.cuda import is_available as cuda_is_available, is_bf16_supported

from model import GPT

import tiktoken


def main():
    parser = ArgumentParser(description="Inference script")

    parser.add_argument("--prompt", default="\n", type=str)
    parser.add_argument("--max_tokens", default=300, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top_k", default=200, type=int)
    parser.add_argument("--checkpoint_path", default="./out/ckpt.pt", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    dtype = (
        torch.bfloat16
        if args.device == "cuda" and is_bf16_supported()
        else torch.float32
    )

    forward_context = autocast(device_type=args.device, dtype=dtype)

    torch.set_float32_matmul_precision("high")

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    checkpoint = torch.load(
        args.checkpoint_path, map_location=args.device, weights_only=True
    )

    model = GPT(**checkpoint["model_args"]).to(args.device)

    print("Compiling model")
    model = torch.compile(model)

    model.load_state_dict(checkpoint["model"])

    print("Model checkpoint loaded successfully")

    tokenizer = tiktoken.get_encoding("r50k_base")

    start_ids = tokenizer.encode(args.prompt, allowed_special={"<|endoftext|>"})

    prompts = torch.tensor(start_ids, dtype=torch.long, device=args.device).unsqueeze(0)

    model.eval()

    print("Generating ...")

    with forward_context:
        out = model.generate(prompts, args.max_tokens, args.temperature, args.top_k)

    out = tokenizer.decode(out.squeeze(0).tolist())

    print(out)


if __name__ == "__main__":
    main()
