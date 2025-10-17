import random

from functools import partial
from argparse import ArgumentParser

import torch

from torch.utils.data import ConcatDataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adafactor
from torch.amp import autocast
from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.text import Perplexity

from src.nope_gpt.model import NoPEGPT
from src.nope_gpt.tokenization import BaseTokenizer, ChatTokenizer

from data import SmolTalk, UltraFeedbackSFT, pad_collate, IGNORE_INDEX

from tqdm import tqdm

DATASET_SUBSETS = SmolTalk.SUBSETS | {"ultra-feedback"}


def main():
    parser = ArgumentParser(
        description="Instruction-tune and align the GPT using supervised training."
    )

    csv_list = partial(str.split, sep=",")

    parser.add_argument("--base_checkpoint_path", required=True, type=str)
    parser.add_argument("--dataset_subsets", default=["all"], type=csv_list)
    parser.add_argument("--max_tokens_per_sample", default=8192, type=int)
    parser.add_argument("--filter_long_samples", action="store_true")
    parser.add_argument("--num_dataset_processes", default=8, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=128, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--low_memory_optimizer", action="store_true")
    parser.add_argument("--max_gradient_norm", default=1.0, type=float)
    parser.add_argument("--num_epochs", default=2, type=int)
    parser.add_argument("--lora_rank", default=8, type=int)
    parser.add_argument("--lora_alpha", default=2.0, type=float)
    parser.add_argument("--freeze_token_embeddings", action="store_true")
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--eval_interval", default=1, type=int)
    parser.add_argument("--num_eval_samples", default=2048, type=int)
    parser.add_argument("--checkpoint_interval", default=1, type=int)
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_dir_path", default="./runs", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if len(args.dataset_subsets) == 0:
        raise ValueError("No dataset subsets provided.")

    for subset in args.dataset_subsets:
        if subset not in DATASET_SUBSETS:
            raise ValueError(f"Invalid dataset subset, {subset} given.")

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

    checkpoint = torch.load(args.base_checkpoint_path, map_location=args.device)

    tokenizer_args = checkpoint["tokenizer_args"]

    tokenizer = BaseTokenizer.from_tiktoken(**tokenizer_args)

    tokenizer = ChatTokenizer(tokenizer)

    datasets = []

    for subset in frozenset(args.dataset_subsets):
        if subset in SmolTalk.SUBSETS:
            dataset = SmolTalk(
                tokenizer,
                subset=subset,
                max_tokens_per_sample=args.max_tokens_per_sample,
                filter_long_samples=args.filter_long_samples,
                num_processes=args.num_dataset_processes,
            )

            datasets.append(dataset)

        if subset == "ultra-feedback":
            dataset = UltraFeedbackSFT(
                tokenizer,
                split="train",
                max_tokens_per_sample=args.max_tokens_per_sample,
                filter_long_samples=args.filter_long_samples,
                num_processes=args.num_dataset_processes,
            )

            datasets.append(dataset)

    dataset = ConcatDataset(datasets)

    n_train_samples = len(dataset) - args.num_eval_samples

    training, testing = random_split(dataset, [n_train_samples, args.num_eval_samples])

    right_pad_collate = partial(
        pad_collate,
        padding_side="right",
        padding_index=tokenizer.pad_token,
    )

    new_dataloader = partial(
        DataLoader,
        collate_fn=right_pad_collate,
        batch_size=args.batch_size,
        pin_memory="cpu" not in args.device,
    )

    train_loader = new_dataloader(training, shuffle=True)
    test_loader = new_dataloader(testing)

    model_args = checkpoint["model_args"]

    model = NoPEGPT(**model_args)

    state_dict = checkpoint["model"]

    # Compensate for compiled state dict.
    for key in list(state_dict.keys()):
        state_dict[key.replace("_orig_mod.", "")] = state_dict.pop(key)

    model.load_state_dict(state_dict)

    model.freeze_model_parameters()

    if not args.freeze_token_embeddings:
        model.unfreeze_token_embeddings()

    model.resize_token_embeddings(tokenizer.vocabulary_size)

    lora_args = {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
    }

    model.add_lora_adapters(**lora_args)

    print(f"Model has {model.num_trainable_params:,} trainable parameters")

    if args.activation_checkpointing:
        model.decoder.enable_activation_checkpointing()

    model = model.to(args.device)

    optimizer = Adafactor(
        model.parameters(),
        lr=args.learning_rate,
        foreach=not args.low_memory_optimizer,
    )

    starting_epoch = 1

    if args.resume:
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        starting_epoch += checkpoint["epoch"]

        print("Previous checkpoint resumed successfully")

    model.train()

    loss_function = CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    perplexity_metric = Perplexity(ignore_index=IGNORE_INDEX).to(args.device)

    for epoch in range(starting_epoch, args.num_epochs + 1):
        total_cross_entropy, total_gradient_norm = 0.0, 0.0
        total_batches, total_steps = 0, 0

        for step, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1
        ):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            with amp_context:
                y_pred = model.forward(x)

                # Flatten the batch and time dimensions.
                y_pred = y_pred.view(-1, y_pred.size(-1))
                y = y.view(-1)

                loss = loss_function(y_pred, y)

                scaled_loss = loss / args.gradient_accumulation_steps

            scaled_loss.backward()

            total_cross_entropy += loss.item()
            total_batches += 1

            if step % args.gradient_accumulation_steps == 0:
                norm = clip_grad_norm_(model.parameters(), args.max_gradient_norm)

                optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                total_gradient_norm += norm.item()
                total_steps += 1

        average_cross_entropy = total_cross_entropy / total_batches
        average_gradient_norm = total_gradient_norm / total_steps

        logger.add_scalar("Cross Entropy", average_cross_entropy, epoch)
        logger.add_scalar("Gradient Norm", average_gradient_norm, epoch)

        print(
            f"Epoch {epoch}:",
            f"Cross Entropy: {average_cross_entropy:.5f},",
            f"Gradient Norm: {average_gradient_norm:.4f}",
        )

        if epoch % args.eval_interval == 0:
            model.eval()

            for x, y in tqdm(test_loader, desc="Testing", leave=False):
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)

                with torch.no_grad():
                    y_pred = model.forward(x)

                perplexity_metric.update(y_pred, y)

            perplexity = perplexity_metric.compute()

            logger.add_scalar("Perplexity", perplexity, epoch)

            print(f"Perplexity: {perplexity:.3f}")

            perplexity_metric.reset()

            model.train()

        if epoch % args.checkpoint_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "tokenizer_args": tokenizer_args,
                "model_args": model_args,
                "lora_args": lora_args,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")

    print("Done!")


if __name__ == "__main__":
    main()
