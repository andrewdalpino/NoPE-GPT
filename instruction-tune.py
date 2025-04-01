import random
from functools import partial

from argparse import ArgumentParser

import torch

from torch.utils.data import DataLoader
from torch.optim import Adafactor
from torch.amp import autocast
from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.text import Perplexity

from model import LightGPT
from data import SmolTalk, left_pad_collate

from tiktoken import Encoding

from tqdm import tqdm


def main():
    parser = ArgumentParser(description="Instruction fine-tune the GPT.")

    parser.add_argument(
        "--base_checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--dataset_subset", default="all", choices=SmolTalk.SUBSETS)
    parser.add_argument("--num_dataset_processes", default=2, type=int)
    parser.add_argument("--max_tokens_per_sample", default=1048, type=int)
    parser.add_argument("--train_on_inputs", action="store_true")
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=64, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--rms_decay", default=-0.8, type=float)
    parser.add_argument("--low_memory_optimizer", action="store_true")
    parser.add_argument("--max_gradient_norm", default=1.0, type=float)
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--rank", default=8, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--dropout", default=0.05, type=float)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--eval_interval", default=1, type=int)
    parser.add_argument("--checkpoint_interval", default=1, type=int)
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/instruct.pt", type=str
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_dir_path", default="./runs/instruction-tune", type=str)
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

    amp_context = autocast(device_type=args.device, dtype=dtype)

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    logger = SummaryWriter(args.run_dir_path)

    checkpoint = torch.load(
        args.base_checkpoint_path, map_location=args.device, weights_only=False
    )

    tokenizer = checkpoint["tokenizer"]

    padding_index = tokenizer.n_vocab
    im_start_index = tokenizer.n_vocab + 1
    im_end_index = tokenizer.n_vocab + 2

    tokenizer = Encoding(
        name=tokenizer.name,
        pat_str=tokenizer._pat_str,
        mergeable_ranks=tokenizer._mergeable_ranks,
        special_tokens={
            **tokenizer._special_tokens,
            "<|pad|>": padding_index,
            "<|im_start|>": im_start_index,
            "<|im_end|>": im_end_index,
        },
    )

    dataset = SmolTalk(
        tokenizer,
        subset=args.dataset_subset,
        max_tokens_per_sample=args.max_tokens_per_sample,
        train_on_inputs=args.train_on_inputs,
        padding_index=padding_index,
    )

    _, training, testing, _ = random_split(dataset, (0.0, 0.09, 0.01, 0.9))

    collate = partial(left_pad_collate, padding_index=padding_index)

    train_loader = DataLoader(
        training,
        collate_fn=collate,
        batch_size=args.batch_size,
        pin_memory="cpu" not in args.device,
        shuffle=True,
        num_workers=args.num_dataset_processes,
    )
    test_loader = DataLoader(
        testing,
        collate_fn=collate,
        batch_size=args.batch_size,
        pin_memory="cpu" not in args.device,
        shuffle=False,
        num_workers=args.num_dataset_processes,
    )

    model_args = checkpoint["model_args"]

    model = LightGPT(**model_args)

    if args.activation_checkpointing:
        model.enable_activation_checkpointing()

    state_dict = checkpoint["model"]

    # Compensate for poorly designed compiled state dicts.
    for key in list(state_dict.keys()):
        state_dict[key.replace("_orig_mod.", "")] = state_dict.pop(key)

    model.load_state_dict(state_dict)

    print("Model checkpoint loaded")

    lora_args = {
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
    }

    model.freeze_model_parameters()
    model.resize_token_embeddings(tokenizer.n_vocab)
    model.set_padding_token_index(padding_index)
    model.unfreeze_token_embeddings()
    model.add_lora_parameters(**lora_args)

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
            args.checkpoint_path, map_location=args.device, weights_only=False
        )

        model.token_embeddings.load_state_dict(checkpoint["token_embeddings"])
        model.load_state_dict(checkpoint["lora"], strict=False)

        optimizer.load_state_dict(checkpoint["optimizer"])

        starting_epoch += checkpoint["epoch"]

        print("Previous checkpoint resumed successfully")

    model.train()

    print(f"Model has {model.num_trainable_params:,} trainable parameters")

    perplexity_metric = Perplexity(ignore_index=padding_index).to(args.device)

    print("Instruction-tuning ...")

    for epoch in range(starting_epoch, args.num_epochs + 1):
        total_cross_entropy, total_gradient_norm = 0.0, 0.0
        total_batches, total_steps = 0, 0

        for step, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1
        ):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            with amp_context:
                y_pred, loss = model(x, y)

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
            f"Epoch {epoch}:" f"Cross Entropy: {average_cross_entropy:.5f},",
            f"Gradient Norm: {average_gradient_norm:.4f}",
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

            logger.add_scalar("Perplexity", perplexity, epoch)

            print(f"Perplexity: {perplexity:.3f}")

            perplexity_metric.reset()

            model.train()

        if epoch % args.checkpoint_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "tokenizer": tokenizer,
                "token_embeddings": model.token_embeddings.state_dict(),
                "lora_args": lora_args,
                "lora": model.lora_state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")

    print("Done!")


if __name__ == "__main__":
    main()
