import random

from math import isnan

from argparse import ArgumentParser

import torch

from torch.utils.data import DataLoader
from torch.optim import Adafactor
from torch.amp import autocast
from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.utils.data import random_split

from torchmetrics.text import Perplexity

from model import GPT, GPTWithLoRA
from data import IMDB

import tiktoken

from tqdm import tqdm


def main():
    parser = ArgumentParser(
        description="Fine-tune the foundation model for sentiment analysis."
    )

    parser.add_argument("--base_model_path", default="./out/checkpoint.pt", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=128, type=int)
    parser.add_argument("--learning_rate", default=5e-3, type=float)
    parser.add_argument("--mask_input", default=True, type=bool)
    parser.add_argument("--rank", default=8, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--dropout", default=0.05, type=float)
    parser.add_argument("--num_epochs", default=4, type=int)
    parser.add_argument("--eval_interval", default=1, type=int)
    parser.add_argument("--checkpoint_interval", default=1, type=int)
    parser.add_argument(
        "--checkpoint_path", default="./out/lora_sentiment.pt", type=str
    )
    parser.add_argument("--keep_checkpoint_history", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    torch.set_float32_matmul_precision("high")

    dtype = (
        torch.bfloat16
        if "cuda" in args.device and is_bf16_supported()
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

    training = IMDB(model_args["block_size"], train=True, mask_input=args.mask_input)
    testing = IMDB(model_args["block_size"], train=False, mask_input=args.mask_input)

    train_loader = DataLoader(
        training,
        collate_fn=training.collate,
        batch_size=args.batch_size,
        pin_memory="cpu" not in args.device,
        shuffle=True,
    )
    test_loader = DataLoader(
        testing,
        collate_fn=testing.collate,
        batch_size=args.batch_size,
        pin_memory="cpu" not in args.device,
        shuffle=False,
    )

    model = GPT(**model_args)

    model = torch.compile(model)

    model.load_state_dict(checkpoint["model"])

    print("Model checkpoint loaded")

    lora_args = {
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
    }

    model = GPTWithLoRA(model, **lora_args).to(args.device)

    print("Compiling model")
    model.compile()

    print(f"Model has {model.num_trainable_params:,} trainable parameters")

    optimizer = Adafactor(model.parameters(), lr=args.learning_rate)

    perplexity_metric = Perplexity(ignore_index=training.PADDING_INDEX).to(args.device)

    starting_epoch = 1

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path,
            map_location=args.device,
            weights_only=True,
        )

        model.load_state_dict(checkpoint["lora"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        starting_epoch += checkpoint["epoch"]

        print("Previous checkpoint resumed successfully")

    model.train()

    print("Sentiment-tuning ...")

    for epoch in range(starting_epoch, args.num_epochs + 1):
        total_cross_entropy, total_batches = 0.0, 0

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
                optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            total_batches += 1

        average_cross_entropy = total_cross_entropy / total_batches

        print(
            f"Epoch {epoch}: Cross Entropy: {average_cross_entropy:.5f}",
        )

        if epoch % args.eval_interval == 0:
            model.eval()

            for x, y in tqdm(test_loader, desc="Testing", leave=False):
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)

                with torch.no_grad():
                    y_pred, _ = model(x)

                perplexity_metric.update(y_pred, y)

            perplexity = perplexity_metric.compute()

            print(f"Perplexity: {perplexity:.3f}")

            perplexity_metric.reset()

            model.train()

        if epoch % args.checkpoint_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "lora_args": lora_args,
                "lora": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")


if __name__ == "__main__":
    main()
