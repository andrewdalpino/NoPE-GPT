import random

from os import path
from copy import deepcopy

from datasets import load_dataset

from tiktoken import Encoding

import numpy as np

import torch

from torch import Tensor
from torch.utils.data import IterableDataset, Dataset
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm


class Fineweb(IterableDataset):
    DATASET_NAME = "HuggingFaceFW/fineweb"

    NUM_SHARDS = 1024

    PADDING_INDEX = -100

    def __init__(
        self,
        tokenizer: Encoding,
        root_path: str = "./dataset",
        subset: str | None = "sample-10BT",
        split: str = "train",
        tokens_per_sample: int = 1024,
        samples_per_epoch: int = 4096,
        num_processes: int = 8,
    ):
        super().__init__()

        if subset != None:
            if subset not in ("sample-10BT", "sample-100BT", "sample-350BT"):
                raise ValueError(f"Invalid subset, {subset} given.")

        if split not in ("train", "test"):
            raise ValueError(f"Split must be either train or test, {split} given.")

        if tokens_per_sample < 1:
            raise ValueError(f"Tokens per sample must be greater than 0.")

        if samples_per_epoch < 1:
            raise ValueError(f"Samples per epoch must be greater than 0.")

        dataset_name = f"fineweb-{subset}" if subset != None else "fineweb"

        train_path = path.join(root_path, f"{dataset_name}-train-{tokenizer.name}.bin")
        test_path = path.join(root_path, f"{dataset_name}-test-{tokenizer.name}.bin")

        bin_file_path = train_path if split == "train" else test_path

        self.tokenizer = tokenizer

        if not path.exists(bin_file_path):
            print("Preprocessing dataset ...")

            tokenized_splits = (
                load_dataset(
                    self.DATASET_NAME,
                    name=subset,
                    num_proc=num_processes,
                    split="train",
                )
                .map(
                    self.tokenize,
                    desc="Tokenizing",
                    remove_columns=["text", "token_count"],
                    num_proc=num_processes,
                )
                .train_test_split(
                    test_size=int(2 * samples_per_epoch),
                    shuffle=True,
                )
            )

            for split, dataset in tokenized_splits.items():
                bin_path = train_path if split == "train" else test_path

                total_length = np.sum(dataset["length"], dtype=np.uint64)

                bin_out = np.memmap(
                    bin_path, dtype=np.uint16, mode="w+", shape=total_length
                )

                index = 0

                for i in tqdm(range(self.NUM_SHARDS), desc="Saving"):
                    batch = dataset.shard(
                        num_shards=self.NUM_SHARDS, index=i, contiguous=True
                    ).with_format("numpy")

                    token_batch = np.concatenate(batch["tokens"])

                    n = len(token_batch)

                    bin_out[index : index + n] = token_batch

                    index += n

                bin_out.flush()

        memmap = np.memmap(bin_file_path, dtype=np.uint16, mode="r")

        self.memmap = memmap
        self.max_start = len(memmap) - (tokens_per_sample + 1)
        self.tokens_per_sample = tokens_per_sample
        self.samples_per_epoch = samples_per_epoch

    def tokenize(self, sample: dict) -> dict:
        tokens = self.tokenizer.encode_ordinary(sample["text"])

        tokens.append(self.tokenizer.eot_token)

        return {
            "tokens": tokens,
            "length": len(tokens),
        }

    def __iter__(self):
        for i in range(self.samples_per_epoch):
            start = random.randint(0, self.max_start)
            end = start + self.tokens_per_sample

            x = self.memmap[start:end]
            y = self.memmap[start + 1 : end + 1]

            x = x.astype(np.int64)
            y = y.astype(np.int64)

            assert x.shape == y.shape, "Sample / label shape mismatch."

            yield x, y


class SmolTalk(Dataset):
    DATASET_NAME = "HuggingFaceTB/smoltalk"

    PADDING_INDEX = -100

    PROMPT_TEMPLATE = "<|im_start|>{role}\n{message}\n<|im_end|>"

    DEFAULT_SYSTEM_MESSAGE = PROMPT_TEMPLATE.format(
        role="system",
        message=(
            "You are a knowledgeable and friendly AI assistant named Bonnie. "
            "Your role is to help users by answering their questions, providing information, and offering guidance to the best of your abilities. "
            "When responding, use a warm and professional tone, and break down complex topics into easy-to-understand explanations. "
            "If you are unsure about an answer, it's okay to say you don't know rather than guessing."
        ),
    )

    def __init__(
        self,
        tokenizer: Encoding,
        subset: str = "all",
        max_tokens_per_sample: int = 1024,
    ):
        super().__init__()

        if subset not in ("all", "smol-magpie-ultra"):
            raise ValueError(f"Invalid subset, {subset} given.")

        if max_tokens_per_sample < 1:
            raise ValueError(
                f"Max tokens per sample must be greater than 0, {max_tokens_per_sample} given."
            )

        special_tokens = {
            "<|im_start|>": tokenizer.n_vocab,
            "<|im_end|>": tokenizer.n_vocab + 1,
        }

        self.tokenizer = Encoding(
            name=tokenizer.name,
            pat_str=tokenizer._pat_str,
            mergeable_ranks=tokenizer._mergeable_ranks,
            special_tokens={**tokenizer._special_tokens, **special_tokens},
        )

        self.dataset = load_dataset(self.DATASET_NAME, subset, split="train")

        self.max_tokens_per_sample = max_tokens_per_sample

    def collate(self, batch: list) -> tuple[Tensor, Tensor]:
        """Custom collate function adds left padding to batched samples."""

        sample, labels = [], []

        for x, y in batch:
            sample.append(x)
            labels.append(y)

        x = pad_sequence(
            sample,
            batch_first=True,
            padding_value=self.PADDING_INDEX,
            padding_side="left",
        )
        y = pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.PADDING_INDEX,
            padding_side="left",
        )

        assert x.shape == y.shape, "Sample / label batch shape mismatch."

        return x, y

    def __getitem__(self, index: int):
        row = self.dataset[index]

        text = self.DEFAULT_SYSTEM_MESSAGE

        for message in row["messages"]:
            text += "\n\n"

            text += self.PROMPT_TEMPLATE.format(
                role=message["role"],
                message=message["content"],
            )

        tokens = self.tokenizer.encode_ordinary(text)

        tokens.append(self.tokenizer.eot_token)

        sample = deepcopy(tokens)
        labels = deepcopy(tokens)

        end = min(len(sample), self.max_tokens_per_sample + 1)

        x = torch.tensor(sample[0 : end - 1], dtype=torch.int64)
        y = torch.tensor(labels[1:end], dtype=torch.int64)

        assert x.shape == y.shape, "Sample / label shape mismatch."

        return x, y

    def __len__(self):
        return len(self.dataset)
