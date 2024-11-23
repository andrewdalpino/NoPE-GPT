import random

from argparse import ArgumentParser
from functools import partial

import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast
from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import random_split

from torchmetrics.text import Perplexity

from model import GPT, GPTWithLoRA
from data import Alpaca

import tiktoken

from tqdm import tqdm


def main():
    parser = ArgumentParser(description="Fine-tune the foundation model.")

    parser.add_argument("--base_model_path", default="./out/checkpoint.pt", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--rank", default=8, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--max_gradient_norm", default=1.0, type=float)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--eval_epochs", default=1, type=int)
    parser.add_argument("--checkpoint_epochs", default=2, type=int)
    parser.add_argument("--checkpoint_path", default="./out/lora.pt", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    torch.set_float32_matmul_precision("high")

    dtype = (
        torch.bfloat16
        if args.device == "cuda" and is_bf16_supported()
        else torch.float32
    )

    forward_context = autocast(device_type=args.device, dtype=dtype)

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    checkpoint = torch.load(
        args.base_model_path, map_location=args.device, weights_only=True
    )

    model_args = checkpoint["model_args"]

    dataset = Alpaca(max_tokens_per_sample=model_args["block_size"])

    training, testing = random_split(dataset, [0.9, 0.1])

    train_loader = DataLoader(
        training,
        collate_fn=dataset.collate,
        batch_size=args.batch_size,
        pin_memory=True,
    )

    test_loader = DataLoader(
        testing,
        collate_fn=dataset.collate,
        batch_size=args.batch_size,
        pin_memory=True,
    )

    model = GPT(**model_args)

    model = torch.compile(model)

    model.load_state_dict(checkpoint["model"])

    print("Model checkpoint loaded")

    lora_args = {
        "rank": args.rank,
        "alpha": args.alpha,
    }

    model = GPTWithLoRA(model, **lora_args).to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, fused=True)

    perplexity_metric = Perplexity(ignore_index=dataset.PADDING_INDEX).to(args.device)

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path,
            map_location=args.device,
            weights_only=True,
        )

        model.load_state_dict(checkpoint["lora"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])

        print("Previous checkpoint resumed successfully")

    print("Compiling model")
    model.compile()

    print(f"Model has {model.num_trainable_params:,} trainable parameters")

    model.train()

    print("Fine-tuning ...")

    for epoch in range(1, args.num_epochs + 1):
        total_cross_entropy, total_gradient_norm = 0.0, 0.0
        total_batches, total_steps = 0, 0

        for step, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1
        ):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            with forward_context:
                y_pred, loss = model(x, y)

                scaled_loss = loss / args.gradient_accumulation_steps

                scaled_loss.backward()

            total_cross_entropy += loss.item()

            if step % args.gradient_accumulation_steps == 0:
                norm = clip_grad_norm_(model.parameters(), args.max_gradient_norm)

                optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                total_gradient_norm += norm.item()
                total_steps += 1

            total_batches += 1

        average_cross_entropy = total_cross_entropy / total_batches
        average_gradient_norm = total_gradient_norm / total_steps

        print(
            f"Epoch {epoch}:",
            f"Cross Entropy: {average_cross_entropy:.5f},",
            f"Gradient Norm: {average_gradient_norm:.4f}",
        )

        if epoch % args.eval_epochs == 0:
            model.eval()

            for x, y in tqdm(test_loader, desc="Testing", leave=False):
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)

                with torch.no_grad():
                    with forward_context:
                        y_pred, _ = model(x)

                    perplexity_metric.update(y_pred, y)

            perplexity = perplexity_metric.compute().item()

            print(f"Perplexity: {perplexity:.4f}")

            perplexity_metric.reset()

            model.train()

        if epoch % args.checkpoint_epochs == 0:
            checkpoint = {
                "lora": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lora_args": lora_args,
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")


if __name__ == "__main__":
    main()
