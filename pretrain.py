import sys
import random
import signal
import warnings

from os import environ
from argparse import ArgumentParser
from contextlib import nullcontext
from functools import partial

import torch

from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss
from torch.optim import Adafactor
from torch.amp import autocast
from torch.cuda import set_device, is_available as cuda_is_available, is_bf16_supported
from torch.nn.utils import clip_grad_norm_
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from torchdata.stateful_dataloader import StatefulDataLoader

from torchmetrics.text import Perplexity

import tiktoken

from data import Fineweb, IGNORE_INDEX
from src.nope_gpt.model import NoPEGPT

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
    parser.add_argument("--ddp_sharding_level", default=2, choices={0, 2, 3}, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=128, type=int)
    parser.add_argument("--tokens_per_sample", default=4096, type=int)
    parser.add_argument("--max_steps", default=20000, type=int)
    parser.add_argument("--learning_rate", default=1e-2, type=float)
    parser.add_argument("--low_memory_optimizer", action="store_true")
    parser.add_argument("--max_gradient_norm", default=1.0, type=float)
    parser.add_argument("--embedding_dimensions", default=1024, type=int)
    parser.add_argument("--num_q_heads", default=16, type=int)
    parser.add_argument("--num_kv_heads", default=4, type=int)
    parser.add_argument("--num_decoder_layers", default=16, type=int)
    parser.add_argument("--hidden_ratio", default=4, choices={1, 2, 4}, type=int)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--eval_interval", default=100, type=int)
    parser.add_argument("--num_eval_samples", default=2048, type=int)
    parser.add_argument("--checkpoint_interval", default=100, type=int)
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_dir_path", default="./runs", type=str)
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

        if args.seed:
            args.seed += RANK

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    torch.set_float32_matmul_precision("high")

    dtype = (
        torch.bfloat16
        if "cuda" in args.device and is_bf16_supported()
        else torch.float32
    )

    amp_context = autocast(device_type=args.device, dtype=dtype)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    logger = SummaryWriter(args.run_dir_path)

    tokenizer = tiktoken.get_encoding(args.token_encoding)

    dataset = Fineweb(
        root_path=args.dataset_path,
        subset=args.dataset_subset,
        tokenizer=tokenizer,
        tokens_per_sample=args.tokens_per_sample,
    )

    n_train_samples = len(dataset) - args.num_eval_samples

    training, testing = random_split(dataset, [n_train_samples, args.num_eval_samples])

    if IS_DDP:
        sampler = DistributedSampler(
            training,
            num_replicas=WORLD_SIZE,
            rank=RANK,
        )
    else:
        sampler = SequentialSampler(training)

    train_loader = StatefulDataLoader(
        training,
        batch_size=args.batch_size,
        sampler=sampler,
        pin_memory="cpu" not in args.device,
        snapshot_every_n_steps=args.gradient_accumulation_steps,
    )

    test_loader = DataLoader(
        testing,
        batch_size=args.batch_size,
        pin_memory="cpu" not in args.device,
    )

    model_args = {
        "vocabulary_size": tokenizer.n_vocab,
        "embedding_dimensions": args.embedding_dimensions,
        "num_q_heads": args.num_q_heads,
        "num_kv_heads": args.num_kv_heads,
        "num_decoder_layers": args.num_decoder_layers,
        "hidden_ratio": args.hidden_ratio,
        "dropout": args.dropout,
    }

    model = NoPEGPT(**model_args)

    if args.activation_checkpointing:
        model.decoder.enable_activation_checkpointing()

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
        foreach=not args.low_memory_optimizer,
    )

    step = 1

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=args.device, weights_only=False
        )

        train_loader.load_state_dict(checkpoint["train_loader"])

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        step += checkpoint["step"]

        print("Previous checkpoint resumed successfully")

    model.train()

    print(f"Model has {model.num_trainable_params:,} trainable parameters")

    loss_function = CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    perplexity_metric = Perplexity(ignore_index=IGNORE_INDEX).to(args.device)

    register_signal_handlers()

    print("Pretraining ...")

    new_progress_bar = partial(
        tqdm,
        total=args.gradient_accumulation_steps,
        leave=False,
    )

    total_cross_entropy = 0.0

    progress_bar = new_progress_bar(desc=f"Step {step:,}")

    for index, (x, y) in enumerate(train_loader, start=1):
        x = x.to(args.device, non_blocking=True)
        y = y.to(args.device, non_blocking=True)

        with amp_context:
            y_pred = model.forward(x)

            y_pred = y_pred.view(-1, y_pred.size(-1))
            y = y.view(-1)  # Flatten the batch dimension.

            loss = loss_function(y_pred, y)

            scaled_loss = loss / args.gradient_accumulation_steps

        sync_and_step = index % args.gradient_accumulation_steps == 0

        gradient_synchronization_context = (
            model.no_sync() if IS_DDP and not sync_and_step else nullcontext()
        )

        with gradient_synchronization_context:
            scaled_loss.backward()

        total_cross_entropy += loss.item()

        progress_bar.update(1)

        if sync_and_step:
            norm = clip_grad_norm_(model.parameters(), args.max_gradient_norm)

            optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            progress_bar.close()

            if IS_MASTER:
                average_cross_entropy = (
                    total_cross_entropy / args.gradient_accumulation_steps
                )

                gradient_norm = norm.item()

                logger.add_scalar("Cross Entropy", average_cross_entropy, step)
                logger.add_scalar("Gradient Norm", gradient_norm, step)

                print(
                    f"Step {step:,}:",
                    f"Cross Entropy: {average_cross_entropy:.5f},",
                    f"Gradient Norm: {gradient_norm:.4f}",
                )

            if IS_MASTER and step % args.eval_interval == 0:
                model.eval()

                for x, y in tqdm(test_loader, desc="Testing", leave=False):
                    x = x.to(args.device, non_blocking=True)
                    y = y.to(args.device, non_blocking=True)

                    with torch.no_grad():
                        y_pred = model.forward(x)

                    perplexity_metric.update(y_pred, y)

                perplexity = perplexity_metric.compute()

                logger.add_scalar("Perplexity", perplexity, step)

                print(f"Perplexity: {perplexity:.3f}")

                perplexity_metric.reset()

                model.train()

            if IS_MASTER and step % args.checkpoint_interval == 0:
                checkpoint = {
                    "step": step,
                    "tokenizer": tokenizer,
                    "train_loader": train_loader.state_dict(),
                    "model_args": model_args,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }

                torch.save(checkpoint, args.checkpoint_path)

                print("Checkpoint saved")

            if step >= args.max_steps:
                break

            step += 1

            total_cross_entropy = 0.0

            progress_bar = new_progress_bar(desc=f"Step {step:,}")

    if IS_DDP:
        ddp_cleanup()

    progress_bar.close()

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
