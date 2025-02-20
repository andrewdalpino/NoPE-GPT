import sys
import random
import signal
import warnings

from os import environ
from argparse import ArgumentParser
from contextlib import nullcontext
from functools import partial

import torch

from torch.utils.data import DataLoader
from torch.optim import Adafactor
from torch.amp import autocast
from torch.cuda import set_device, is_available as cuda_is_available, is_bf16_supported
from torch.nn.utils import clip_grad_norm_
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.text import Perplexity

import tiktoken

from data import Fineweb
from model import LightGPT

from tqdm import tqdm

RANK = int(environ.get("RANK", -1))
LOCAL_RANK = int(environ.get("LOCAL_RANK", -1))
WORLD_SIZE = int(environ.get("WORLD_SIZE", -1))

IS_DDP = WORLD_SIZE > 1

IS_MASTER = RANK == 0 or not IS_DDP

DDP_BACKEND = "nccl"


def main():
    parser = ArgumentParser(description="Pretrain the GPT.")

    parser.add_argument(
        "--dataset_subset",
        default="sample-10BT",
        choices={"sample-10BT", "sample-100BT", "sample-350BT", None},
    )
    parser.add_argument(
        "--token_encoding",
        default="r50k_base",
        choices={"r50k_base", "p50k_base", "cl100k_base", "o200k_base"},
    )
    parser.add_argument("--dataset_path", default="./datasets", type=str)
    parser.add_argument("--num_dataset_processes", default=8, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=128, type=int)
    parser.add_argument("--tokens_per_sample", default=1024, type=int)
    parser.add_argument("--samples_per_epoch", default=4096, type=int)
    parser.add_argument("--num_epochs", default=1686, type=int)
    parser.add_argument("--learning_rate", default=1e-2, type=float)
    parser.add_argument("--rms_decay", default=-0.8, type=float)
    parser.add_argument("--low_memory_optimizer", action="store_true")
    parser.add_argument("--max_gradient_norm", default=1.0, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--embedding_dimensions", default=1024, type=int)
    parser.add_argument("--num_attention_heads", default=16, type=int)
    parser.add_argument("--num_hidden_layers", default=24, type=int)
    parser.add_argument("--feed_forward_ratio", default=4, choices={1, 2, 4})
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--ddp_sharding_level", default=2, choices={0, 2, 3})
    parser.add_argument("--eval_interval", default=10, type=int)
    parser.add_argument("--checkpoint_interval", default=20, type=int)
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_dir_path", default="./runs/pretrain", type=str)
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

    torch.set_float32_matmul_precision("high")

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    dtype = (
        torch.bfloat16
        if "cuda" in args.device and is_bf16_supported()
        else torch.float32
    )

    amp_context = autocast(device_type=args.device, dtype=dtype)

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    logger = SummaryWriter(args.run_dir_path)

    tokenizer = tiktoken.get_encoding(args.token_encoding)

    build_fineweb = partial(
        Fineweb,
        root_path=args.dataset_path,
        subset=args.dataset_subset,
        tokenizer=tokenizer,
        tokens_per_sample=args.tokens_per_sample,
        samples_per_epoch=args.samples_per_epoch,
    )

    training = build_fineweb(split="train")
    testing = build_fineweb(split="test")

    train_loader = DataLoader(
        training, batch_size=args.batch_size, pin_memory="cpu" not in args.device
    )
    test_loader = DataLoader(
        testing, batch_size=args.batch_size, pin_memory="cpu" not in args.device
    )

    model_args = {
        "vocabulary_size": tokenizer.n_vocab,
        "embedding_dimensions": args.embedding_dimensions,
        "num_heads": args.num_attention_heads,
        "num_layers": args.num_hidden_layers,
        "feed_forward_ratio": args.feed_forward_ratio,
        "dropout": args.dropout,
        "padding_index": training.PADDING_INDEX,
    }

    model = LightGPT(**model_args)

    if args.activation_checkpointing:
        model.enable_activation_checkpointing()

    print("Compiling model")
    model = torch.compile(model)

    if IS_DDP:
        match args.ddp_sharding_level:
            case 0:
                sharding_strategy = ShardingStrategy.NO_SHARD
            case 2:
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
            case 3:
                sharding_strategy = ShardingStrategy.FULL_SHARD

        model = FullyShardedDataParallel(
            model,
            device_id=LOCAL_RANK,
            sharding_strategy=sharding_strategy,
            use_orig_params=True,
        )

    model = model.to(args.device)

    optimizer = Adafactor(
        model.parameters(),
        lr=args.learning_rate,
        beta2_decay=args.rms_decay,
        foreach=not args.low_memory_optimizer,
    )

    starting_epoch = 1

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path, map_location="cpu", weights_only=True
        )  # Always load into CPU RAM first to prevent CUDA out-of-memory errors.

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        starting_epoch += checkpoint["epoch"]

        model = model.to(args.device)

        print("Previous checkpoint resumed successfully")

    model.train()

    print(f"Model has {model.num_trainable_params:,} trainable parameters")

    perplexity_metric = Perplexity(ignore_index=training.PADDING_INDEX).to(args.device)

    register_signal_handlers()

    print("Pretraining ...")

    for epoch in range(starting_epoch, args.num_epochs + 1):
        total_cross_entropy, total_gradient_norm = 0.0, 0.0
        total_batches, total_steps = 0, 0

        for step, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1
        ):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            with amp_context:
                y_pred, loss = model.forward(x, y)

                scaled_loss = loss / args.gradient_accumulation_steps

            sync_and_step = step % args.gradient_accumulation_steps == 0

            gradient_synchronization_context = (
                model.no_sync() if IS_DDP and not sync_and_step else nullcontext()
            )

            with gradient_synchronization_context:
                scaled_loss.backward()

            total_cross_entropy += loss.item()

            if sync_and_step:
                norm = clip_grad_norm_(model.parameters(), args.max_gradient_norm)

                optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                total_gradient_norm += norm.item()
                total_steps += 1

            total_batches += 1

        if IS_MASTER:
            average_cross_entropy = total_cross_entropy / total_batches
            average_gradient_norm = total_gradient_norm / total_steps

            logger.add_scalar("cross entropy", average_cross_entropy, epoch)
            logger.add_scalar("gradient norm", average_gradient_norm, epoch)

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
                    y_pred, _ = model.forward(x, None)

                perplexity_metric.update(y_pred, y)

            perplexity = perplexity_metric.compute()

            logger.add_scalar("perplexity", perplexity, epoch)

            print(f"Perplexity: {perplexity:.3f}")

            perplexity_metric.reset()

            model.train()

        if epoch % args.checkpoint_interval == 0 and IS_MASTER:
            checkpoint = {
                "epoch": epoch,
                "model_args": model_args,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "token_encoding": args.token_encoding,
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")

    if IS_DDP:
        ddp_cleanup()

    print("Done!")


def register_signal_handlers():
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)


def shutdown(signum, frame):
    print("Hold on, attempting to exit gracefully")

    if IS_DDP:
        ddp_cleanup()

    sys.exit(0)


def ddp_cleanup():
    destroy_process_group()


if __name__ == "__main__":
    main()
