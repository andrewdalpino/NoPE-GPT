import sys
import random
import signal
import warnings

from os import path, environ
from argparse import ArgumentParser
from contextlib import nullcontext

import torch

from torch.utils.data import DataLoader
from torch.optim import Adafactor
from torch.amp import autocast
from torch.cuda import set_device, is_available as cuda_is_available, is_bf16_supported
from torch.nn.utils import clip_grad_norm_
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel

from torchmetrics.text import Perplexity

from model import GPT
from data import Openwebtext

from tqdm import tqdm

RANK = int(environ.get("RANK", -1))
LOCAL_RANK = int(environ.get("LOCAL_RANK", -1))
WORLD_SIZE = int(environ.get("WORLD_SIZE", -1))

IS_DDP = WORLD_SIZE > 1

IS_MASTER = RANK == 0 or not IS_DDP

DDP_BACKEND = "nccl"  # nccl, gloo, etc.


def main():
    parser = ArgumentParser(description="Pre-train the GPT.")

    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=128, type=int)
    parser.add_argument("--samples_per_epoch", default=4096, type=int)
    parser.add_argument("--learning_rate", default=5e-3, type=float)
    parser.add_argument("--max_gradient_norm", default=1.0, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--num_epochs", default=2145, type=int)
    parser.add_argument("--block_size", default=1024, type=int)
    parser.add_argument("--embedding_dimensions", default=1024, type=int)
    parser.add_argument("--num_attention_heads", default=16, type=int)
    parser.add_argument("--num_hidden_layers", default=24, type=int)
    parser.add_argument("--eval_interval", default=10, type=int)
    parser.add_argument("--checkpoint_interval", default=20, type=int)
    parser.add_argument("--checkpoint_path", default="./out/checkpoint.pt", type=str)
    parser.add_argument("--dataset_path", default="./dataset", type=str)
    parser.add_argument("--num_dataset_processes", default=8, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(f"Batch size must be greater than 0, {args.batch_size} given.")

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"Gradient accumulation steps must be greater than 0, {args.gradient_accumulation_steps} given."
        )

    if args.learning_rate < 0:
        raise ValueError(
            f"Learning rate must be a positive value, {args.learning_rate} given."
        )

    if args.num_epochs < 1:
        raise ValueError(f"Must train for at least 1 epoch, {args.num_epochs} given.")

    if args.eval_interval < 1:
        raise ValueError(
            f"Eval interval must be greater than 0, {args.eval_interval} given."
        )

    if args.checkpoint_interval < 1:
        raise ValueError(
            f"Checkpoint interval must be greater than 0, {args.checkpoint_interval} given."
        )

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    torch.set_float32_matmul_precision("high")

    if IS_DDP:
        init_process_group(backend=DDP_BACKEND, world_size=WORLD_SIZE)

        args.device = f"cuda:{LOCAL_RANK}"

        set_device(args.device)

        if args.gradient_accumulation_steps % WORLD_SIZE != 0:
            warnings.warn(
                "Number of gradient accumulation steps does not"
                "divide evenly into the world size."
            )

        args.gradient_accumulation_steps //= WORLD_SIZE

        assert (
            args.gradient_accumulation_steps > 0
        ), "World size is larger than the number of gradient accumulation steps."

        if args.samples_per_epoch % WORLD_SIZE != 0:
            warnings.warn(
                "Number of samples per epoch does not"
                "divide evenly into the world size."
            )

        args.samples_per_epoch //= WORLD_SIZE

        assert (
            args.samples_per_epoch > 0
        ), "World size is larger than the number of samples per epoch."

        if args.seed:
            args.seed += RANK

    dtype = (
        torch.bfloat16
        if "cuda" in args.device and is_bf16_supported()
        else torch.float32
    )

    forward_context = autocast(device_type=args.device, dtype=dtype)

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    training = Openwebtext(
        root_path=args.dataset_path,
        train=True,
        tokens_per_sample=args.block_size,
        samples_per_epoch=args.samples_per_epoch,
        num_processes=args.num_dataset_processes,
    )
    testing = Openwebtext(
        root_path=args.dataset_path,
        train=False,
        tokens_per_sample=args.block_size,
        samples_per_epoch=args.samples_per_epoch,
        num_processes=args.num_dataset_processes,
    )

    train_loader = DataLoader(
        training, batch_size=args.batch_size, pin_memory="cpu" not in args.device
    )
    test_loader = DataLoader(
        testing, batch_size=args.batch_size, pin_memory="cpu" not in args.device
    )

    model_args = {
        "block_size": args.block_size,
        "embedding_dimensions": args.embedding_dimensions,
        "num_heads": args.num_attention_heads,
        "num_layers": args.num_hidden_layers,
        "dropout": args.dropout,
        "activation_checkpointing": args.activation_checkpointing,
        "vocabulary_size": training.vocabulary_size,
        "padding_index": training.PADDING_INDEX,
        "eos_index": training.eos_index,
    }

    model = GPT(**model_args).to(args.device)

    if IS_DDP:
        model = DistributedDataParallel(model, device_ids=[LOCAL_RANK])

    print("Compiling model")
    model = torch.compile(model)

    print(f"Model has {model.num_trainable_params:,} trainable parameters")

    if IS_DDP:
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=Adafactor,
            lr=args.learning_rate,
        )
    else:
        optimizer = Adafactor(model.parameters(), lr=args.learning_rate)

    starting_epoch = 1

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=args.device, weights_only=True
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        starting_epoch += checkpoint["epoch"]

        print("Previous checkpoint resumed successfully")

    perplexity_metric = Perplexity(ignore_index=training.PADDING_INDEX).to(args.device)

    model.train()

    signal.signal(signal.SIGTERM, on_sigterm)

    print("Pre-training ...")

    for epoch in range(starting_epoch, args.num_epochs + 1):
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

            sync_and_step = step % args.gradient_accumulation_steps == 0

            backward_context = (
                model.no_sync() if IS_DDP and not sync_and_step else nullcontext()
            )

            with backward_context:
                scaled_loss.backward()

            total_cross_entropy += loss.item()

            if sync_and_step:
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

        if epoch % args.eval_interval == 0 and IS_MASTER:
            model.eval()

            for x, y in tqdm(test_loader, desc="Testing", leave=False):
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)

                with torch.no_grad():
                    with forward_context:
                        y_pred, _ = model(x)

                    perplexity_metric.update(y_pred, y)

            perplexity = perplexity_metric.compute()

            print(f"Perplexity: {perplexity:.3f}")

            perplexity_metric.reset()

            model.train()

        if epoch % args.checkpoint_interval == 0 and IS_MASTER:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "epoch": epoch,
                "seed": args.seed,
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")

    if IS_DDP:
        destroy_process_group()

    print("Done!")


def on_sigterm(signum, frame):
    print("Hold on, attempting to exit gracefully.")

    if IS_DDP:
        destroy_process_group()

    sys.exit(0)


if __name__ == "__main__":
    main()
