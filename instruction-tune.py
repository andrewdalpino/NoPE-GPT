import random

from argparse import ArgumentParser

import torch

from torch.utils.data import DataLoader
from torch.optim import Adafactor
from torch.amp import autocast
from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.text import Perplexity

from model import LightGPT, LightGPTInstruct
from data import SmolTalk

import tiktoken

from tiktoken import Encoding

from tqdm import tqdm


def main():
    parser = ArgumentParser(description="Instruction-tune the GPT.")

    parser.add_argument(
        "--base_model_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument(
        "--dataset_subset",
        default="smol-magpie-ultra",
        choices={
            "smol-magpie-ultra",
            "smol-constraints",
            "smol-rewrite",
            "smol-summarize",
            "all",
        },
    )
    parser.add_argument("--num_dataset_processes", default=4, type=int)
    parser.add_argument("--max_tokens_per_sample", default=1048, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=64, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--rms_decay", default=-0.8, type=float)
    parser.add_argument("--optimizer_low_memory", action="store_true")
    parser.add_argument("--num_epochs", default=4, type=int)
    parser.add_argument("--rank", default=8, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--dropout", default=0.05, type=float)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--eval_interval", default=1, type=int)
    parser.add_argument("--checkpoint_interval", default=1, type=int)
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/lora_instruct.pt", type=str
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
        args.base_model_path, map_location=args.device, weights_only=True
    )

    model_args = checkpoint["model_args"]

    tokenizer = tiktoken.get_encoding(checkpoint["token_encoding"])

    special_tokens = {
        "<|im_start|>": tokenizer.n_vocab,
        "<|im_end|>": tokenizer.n_vocab + 1,
    }

    tokenizer = Encoding(
        name=tokenizer.name,
        pat_str=tokenizer._pat_str,
        mergeable_ranks=tokenizer._mergeable_ranks,
        special_tokens={**tokenizer._special_tokens, **special_tokens},
    )

    dataset = SmolTalk(
        tokenizer,
        subset=args.dataset_subset,
        max_tokens_per_sample=args.max_tokens_per_sample,
    )

    training, testing = random_split(dataset, (0.95, 0.05))

    train_loader = DataLoader(
        training,
        collate_fn=dataset.collate,
        batch_size=args.batch_size,
        pin_memory="cpu" not in args.device,
        shuffle=True,
        num_workers=args.num_dataset_processes,
    )
    test_loader = DataLoader(
        testing,
        collate_fn=dataset.collate,
        batch_size=args.batch_size,
        pin_memory="cpu" not in args.device,
        shuffle=False,
        num_workers=args.num_dataset_processes,
    )

    model = LightGPT(**model_args)

    if args.activation_checkpointing:
        model.enable_activation_checkpointing()

    model = torch.compile(model)

    model.load_state_dict(checkpoint["model"])

    print("Model checkpoint loaded")

    lora_args = {
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
    }

    model = LightGPTInstruct(model, **lora_args).to(args.device)

    print("Compiling model")
    model.compile()

    optimizer = Adafactor(
        model.parameters(),
        lr=args.learning_rate,
        beta2_decay=args.rms_decay,
        foreach=not args.optimizer_low_memory,
    )

    starting_epoch = 1

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=args.device, weights_only=True
        )

        model.load_state_dict(checkpoint["lora"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        starting_epoch += checkpoint["epoch"]

        print("Previous checkpoint resumed successfully")

    model.train()

    print(f"Model has {model.num_trainable_params:,} trainable parameters")

    perplexity_metric = Perplexity(ignore_index=dataset.PADDING_INDEX).to(args.device)

    print("Instruction-tuning ...")

    for epoch in range(starting_epoch, args.num_epochs + 1):
        total_cross_entropy, total_batches = 0.0, 0

        for step, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1
        ):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            with amp_context:
                _, loss = model(x, y)

                scaled_loss = loss / args.gradient_accumulation_steps

            scaled_loss.backward()

            total_cross_entropy += loss.item()

            if step % args.gradient_accumulation_steps == 0:
                optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            total_batches += 1

        average_cross_entropy = total_cross_entropy / total_batches

        logger.add_scalar("cross entropy", average_cross_entropy, epoch)

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

            logger.add_scalar("perplexity", perplexity, epoch)

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

    print("Done!")


if __name__ == "__main__":
    main()
