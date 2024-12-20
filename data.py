import random

from os import path
from copy import deepcopy

from datasets import load_dataset

import tiktoken

from bs4 import BeautifulSoup

import numpy as np

import torch

from torch import Tensor
from torch.utils.data import IterableDataset, Dataset
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm


class Openwebtext(IterableDataset):
    DATASET_NAME = "openwebtext"

    FILE_PREFIX = DATASET_NAME

    TRAIN_FILENAME = f"{FILE_PREFIX}-train.bin"
    TEST_FILENAME = f"{FILE_PREFIX}-test.bin"

    TEST_SPLIT_PROPORTION = 0.005
    NUM_SHARDS = 1024

    ENCODING = "r50k_base"

    PADDING_INDEX = -100

    def __init__(
        self,
        root_path: str,
        train: bool = True,
        tokens_per_sample: int = 1024,
        samples_per_epoch: int = 4096,
        num_processes: int = 8,
    ):
        super().__init__()

        if tokens_per_sample < 1:
            raise ValueError(f"Tokens per sample must be greater than 0.")

        if samples_per_epoch < 1:
            raise ValueError(f"Samples per epoch must be greater than 0.")

        train_path = path.join(root_path, self.TRAIN_FILENAME)
        test_path = path.join(root_path, self.TEST_FILENAME)

        self.tokenizer = tiktoken.get_encoding(self.ENCODING)

        if not path.exists(train_path) or not path.exists(test_path):
            tokenized_splits = (
                load_dataset(self.DATASET_NAME, num_proc=num_processes, split="train")
                .train_test_split(test_size=self.TEST_SPLIT_PROPORTION, shuffle=True)
                .map(
                    self.tokenize,
                    desc="Tokenizing",
                    remove_columns=["text"],
                    num_proc=num_processes,
                )
            )

            for split, dataset in tokenized_splits.items():
                bin_path = path.join(root_path, f"{self.FILE_PREFIX}-{split}.bin")

                total_length = np.sum(dataset["length"], dtype=np.uint64)

                bin_out = np.memmap(
                    bin_path, dtype=np.uint16, mode="w+", shape=total_length
                )

                index = 0

                for i in tqdm(range(self.NUM_SHARDS), desc="Writing"):
                    batch = dataset.shard(
                        num_shards=self.NUM_SHARDS, index=i, contiguous=True
                    ).with_format("numpy")

                    token_batch = np.concatenate(batch["tokens"])

                    n = len(token_batch)

                    bin_out[index : index + n] = token_batch

                    index += n

                bin_out.flush()

        bin_file_path = path.join(
            root_path, self.TRAIN_FILENAME if train else self.TEST_FILENAME
        )

        memmap = np.memmap(bin_file_path, dtype=np.uint16, mode="r")

        self.memmap = memmap
        self.max_start = len(memmap) - (tokens_per_sample + 1)
        self.tokens_per_sample = tokens_per_sample
        self.samples_per_epoch = samples_per_epoch

    @property
    def vocabulary_size(self) -> int:
        return self.tokenizer.max_token_value + 1

    @property
    def eos_index(self) -> int:
        return self.tokenizer.eot_token

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


class Alpaca(Dataset):
    DATASET_NAME = "tatsu-lab/alpaca"

    ENCODING = "r50k_base"

    PADDING_INDEX = -100

    PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )

    PROMPT_TEMPLATE_WITH_INPUT = (
        "Below is an instruction that describes a task, paired with an input "
        "that provides further context. Write a response that appropriately "
        "completes the request.\n\n"
        "### Input:\n{input}\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )

    RESPONSE_TEMPLATE = "{output}"

    def __init__(self, max_tokens_per_sample: int = 1024, mask_input: bool = True):
        super().__init__()

        if max_tokens_per_sample < 1:
            raise ValueError(
                f"Max tokens per sample must be greater than 0, {max_tokens_per_sample} given."
            )

        self.dataset = load_dataset(self.DATASET_NAME, split="train")

        self.tokenizer = tiktoken.get_encoding(self.ENCODING)

        self.max_tokens_per_sample = max_tokens_per_sample
        self.mask_input = mask_input

    @property
    def vocabulary_size(self) -> int:
        return self.tokenizer.max_token_value + 1

    @property
    def eos_index(self) -> int:
        return self.tokenizer.eot_token

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

        has_input = len(row["input"]) > 0

        if has_input:
            text = self.PROMPT_TEMPLATE_WITH_INPUT.format(
                input=row["input"], instruction=row["instruction"]
            )
        else:
            text = self.PROMPT_TEMPLATE.format(instruction=row["instruction"])

        tokens = self.tokenizer.encode_ordinary(text)

        sample = deepcopy(tokens)

        if self.mask_input:
            labels = [self.PADDING_INDEX] * len(tokens)
        else:
            labels = deepcopy(tokens)

        text = self.RESPONSE_TEMPLATE.format(output=row["output"])

        tokens = self.tokenizer.encode_ordinary(text)

        tokens.append(self.tokenizer.eot_token)

        sample.extend(tokens)
        labels.extend(tokens)

        end = min(len(sample), self.max_tokens_per_sample + 1)

        x = torch.tensor(sample[0 : end - 1], dtype=torch.int64)
        y = torch.tensor(labels[1:end], dtype=torch.int64)

        assert x.shape == y.shape, "Sample / label shape mismatch."

        return x, y

    def __len__(self):
        return len(self.dataset)


class IMDB(Dataset):
    DATASET_NAME = "stanfordnlp/imdb"

    ENCODING = "r50k_base"

    PADDING_INDEX = -100

    PROMPT_TEMPLATE = (
        "What is the sentiment of this movie review?\n\n"
        "### Review:\n{review}\n\n"
        "### Sentiment:\n"
    )

    RESPONSE_TEMPLATE = "{sentiment}"

    def __init__(
        self,
        max_tokens_per_sample: int = 1024,
        train: bool = True,
        mask_input: bool = True,
    ):
        super().__init__()

        if max_tokens_per_sample < 1:
            raise ValueError(
                f"Max tokens per sample must be greater than 0, {max_tokens_per_sample} given."
            )

        split = "train" if train else "test"

        self.dataset = load_dataset(self.DATASET_NAME, split=split)

        self.tokenizer = tiktoken.get_encoding(self.ENCODING)

        self.max_tokens_per_sample = max_tokens_per_sample
        self.mask_input = mask_input

    @property
    def vocabulary_size(self) -> int:
        return self.tokenizer.max_token_value + 1

    @property
    def eos_index(self) -> int:
        return self.tokenizer.eot_token

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

        soup = BeautifulSoup(row["text"], "html.parser")

        text = self.PROMPT_TEMPLATE.format(review=soup.get_text())

        tokens = self.tokenizer.encode_ordinary(text)

        tokens = tokens[: self.max_tokens_per_sample - 4]  # Leave room for label tokens

        sample = deepcopy(tokens)

        if self.mask_input:
            labels = [self.PADDING_INDEX] * len(tokens)
        else:
            labels = deepcopy(tokens)

        sentiment = "Positive" if row["label"] == 1 else "Negative"

        text = self.RESPONSE_TEMPLATE.format(sentiment=sentiment)

        tokens = self.tokenizer.encode_ordinary(text)

        tokens.append(self.tokenizer.eot_token)

        sample.extend(tokens)
        labels.extend(tokens)

        end = min(len(sample), self.max_tokens_per_sample + 1)

        x = torch.tensor(sample[0 : end - 1], dtype=torch.int64)
        y = torch.tensor(labels[1:end], dtype=torch.int64)

        assert x.shape == y.shape, "Sample / label shape mismatch."

        return x, y

    def __len__(self):
        return len(self.dataset)
