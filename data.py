import random
from os import path

from datasets import load_dataset

import tiktoken

import numpy as np

import torch

from torch.utils.data import IterableDataset, Dataset
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm


class Openwebtext(IterableDataset):
    DATASET_NAME = "openwebtext"

    FILE_PREFIX = DATASET_NAME

    TRAIN_FILENAME = f"{FILE_PREFIX}-train.bin"
    TEST_FILENAME = f"{FILE_PREFIX}-test.bin"

    TEST_SPLIT_PROPORTION = 0.005

    ENCODING = "r50k_base"

    PADDING_INDEX = 50257
    EOS_INDEX = 50256

    VOCABULARY_SIZE = 50258

    NUM_SHARDS = 1024

    def __init__(
        self,
        root_path: str,
        train: bool = True,
        tokens_per_sample: int = 1024,
        samples_per_epoch: int = 8192,
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
            dataset = load_dataset(
                self.DATASET_NAME, num_proc=num_processes, split="train"
            )

            splits = dataset.train_test_split(
                test_size=self.TEST_SPLIT_PROPORTION, shuffle=True
            )

            splits = splits.map(
                self.tokenize,
                desc="Tokenizing",
                remove_columns=["text"],
                num_proc=num_processes,
            )

            for split, dataset in splits.items():
                filename = path.join(root_path, f"{self.FILE_PREFIX}-{split}.bin")

                total_length = np.sum(dataset["length"], dtype=np.uint64)

                bin_out = np.memmap(
                    filename, dtype=np.uint16, mode="w+", shape=total_length
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

        self.tokens_per_sample = tokens_per_sample
        self.samples_per_epoch = samples_per_epoch

        self.bin_file_path = path.join(
            root_path, self.TRAIN_FILENAME if train else self.TEST_FILENAME
        )

    def tokenize(self, sample: dict) -> dict:
        tokens = self.tokenizer.encode_ordinary(sample["text"])

        tokens.append(self.EOS_INDEX)

        return {
            "tokens": tokens,
            "length": len(tokens),
        }

    def __iter__(self):
        data = np.memmap(self.bin_file_path, dtype=np.uint16, mode="r")

        max_start = len(data) - (self.tokens_per_sample + 1)

        for i in range(self.samples_per_epoch):
            start = random.randint(0, max_start)
            end = start + self.tokens_per_sample

            x = data[start:end]
            y = data[start + 1 : end + 1]

            x = x.astype(np.int64)
            y = y.astype(np.int64)

            assert x.shape == y.shape, "Sample / label shape mismatch."

            yield x, y


class Alpaca(Dataset):
    DATASET_NAME = "tatsu-lab/alpaca"

    ENCODING = "r50k_base"

    PADDING_INDEX = 50257
    EOS_INDEX = 50256

    VOCABULARY_SIZE = 50258

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

    def __init__(
        self,
        max_tokens_per_sample: int = 1024,
    ):
        super().__init__()

        if max_tokens_per_sample < 1:
            raise ValueError(
                f"Max tokens per sample must be greater than 0, {max_tokens_per_sample} given."
            )

        self.dataset = load_dataset(self.DATASET_NAME, split="train")

        self.tokenizer = tiktoken.get_encoding(self.ENCODING)

        self.max_tokens_per_sample = max_tokens_per_sample

    @classmethod
    def collate(cls, batch: list):
        """Custom collate function adds left padding to batched samples."""

        samples, labels = [], []

        for x, y in batch:
            samples.append(x)
            labels.append(y)

        x = pad_sequence(
            samples,
            batch_first=True,
            padding_value=cls.PADDING_INDEX,
            padding_side="left",
        )
        y = pad_sequence(
            labels,
            batch_first=True,
            padding_value=cls.PADDING_INDEX,
            padding_side="left",
        )

        assert x.shape == y.shape, "Sample / label batch shape mismatch."

        return x, y

    def __getitem__(self, index: int):
        row = self.dataset[index]

        has_input = len(row["input"]) > 0

        if not has_input:
            text = self.PROMPT_TEMPLATE.format(instruction=row["instruction"])

        else:
            text = self.PROMPT_TEMPLATE_WITH_INPUT.format(
                input=row["input"], instruction=row["instruction"]
            )

        tokens = self.tokenizer.encode_ordinary(text)

        sample = tokens
        label = [self.PADDING_INDEX] * len(tokens)

        tokens = self.tokenizer.encode_ordinary(row["output"])

        tokens.append(self.EOS_INDEX)

        sample.extend(tokens)
        label.extend(tokens)

        end = min(len(sample), self.max_tokens_per_sample)

        x = torch.tensor(sample[0 : end - 1], dtype=torch.int64)
        y = torch.tensor(label[1:end], dtype=torch.int64)

        assert x.shape == y.shape, "Sample / label shape mismatch."

        return x, y

    def __len__(self):
        return len(self.dataset)
