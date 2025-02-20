import random

from os import path
from copy import deepcopy
from functools import partial

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

    PADDING_INDEX = -100

    def __init__(
        self,
        tokenizer: Encoding,
        root_path: str = "./dataset",
        subset: str | None = "sample-10BT",
        split: str = "train",
        tokens_per_sample: int = 1024,
        samples_per_epoch: int = 4096,
    ):
        super().__init__()

        if subset != None:
            if subset not in {"sample-10BT", "sample-100BT", "sample-350BT"}:
                raise ValueError(f"Invalid subset, {subset} given.")

        if split not in {"train", "test"}:
            raise ValueError(f"Split must be either train or test, {split} given.")

        if tokens_per_sample < 1:
            raise ValueError(f"Tokens per sample must be greater than 0.")

        if samples_per_epoch < 1:
            raise ValueError(f"Samples per epoch must be greater than 0.")

        self.tokenizer = tokenizer

        dataset_name = f"fineweb-{subset}" if subset != None else "fineweb"

        train_path = path.join(root_path, f"{dataset_name}-train-{tokenizer.name}.bin")
        test_path = path.join(root_path, f"{dataset_name}-test-{tokenizer.name}.bin")

        bin_file_path = train_path if split == "train" else test_path

        if not path.exists(bin_file_path):
            tokenized_dataset = load_dataset(
                self.DATASET_NAME,
                name=subset,
                split="train",
                streaming=True,
            ).map(
                self.tokenize,
            )

            test = tokenized_dataset.take(4 * samples_per_epoch)
            train = tokenized_dataset.skip(4 * samples_per_epoch)

            tokenized_splits = {
                "test": test,
                "train": train,
            }

            for split, dataset in tokenized_splits.items():
                bin_path = train_path if split == "train" else test_path

                bin_out = np.memmap(bin_path, dtype=np.uint16, mode="w+", shape=1024)

                new_memmap = partial(np.memmap, bin_path, dtype=np.uint16, mode="r+")

                total_length = 0

                for row in tqdm(dataset, desc=f"Writing {split} dataset"):
                    tokens = row["tokens"]

                    length = len(tokens)

                    l_hat = total_length + length

                    delta = l_hat - len(bin_out)

                    if delta > 0:
                        new_bin_out = new_memmap(shape=len(bin_out) + delta)

                        new_bin_out[: len(bin_out)] = bin_out

                        bin_out = new_bin_out

                    bin_out[total_length:l_hat] = tokens

                    total_length = l_hat

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
        }

    def __iter__(self):
        for _ in range(self.samples_per_epoch):
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

    PROMPT_TEMPLATE = "<|im_start|>{role}\n{message}\n<|im_end|>\n"

    REWRITE_SYSTEM_MESSAGE = PROMPT_TEMPLATE.format(
        role="system",
        message=(
            "You're an AI assistant for text re-writing. "
            "Rewrite the input text to make it more concise while preserving its core meaning."
        ),
    )

    def __init__(
        self,
        tokenizer: Encoding,
        subset: str = "smol-magpie-ultra",
        max_tokens_per_sample: int = 1024,
    ):
        super().__init__()

        if subset not in {
            "smol-magpie-ultra",
            "smol-constraints",
            "smol-rewrite",
            "smol-summarize",
            "all",
        }:
            raise ValueError(f"Invalid subset, {subset} given.")

        if max_tokens_per_sample < 1:
            raise ValueError(
                f"Max tokens per sample must be greater than 0, {max_tokens_per_sample} given."
            )

        self.tokenizer = tokenizer

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

        text = ""

        for message in row["messages"]:
            text += self.PROMPT_TEMPLATE.format(
                role=message["role"],
                message=message["content"],
            )

        tokens = self.tokenizer.encode_ordinary(text)

        tokens = tokens[: self.max_tokens_per_sample + 1]

        sample = deepcopy(tokens[:-1])
        labels = deepcopy(tokens[1:])

        x = torch.tensor(sample, dtype=torch.int64)
        y = torch.tensor(labels, dtype=torch.int64)

        assert x.shape == y.shape, "Sample / label shape mismatch."

        return x, y

    def __len__(self):
        return len(self.dataset)
