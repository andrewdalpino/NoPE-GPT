import random
from os import path

from datasets import load_dataset

import tiktoken

import numpy as np

import torch

from torch.utils.data import IterableDataset, Dataset
from torch.nn.functional import pad

from tqdm import tqdm


class Openwebtext(IterableDataset):
    DATASET_NAME = "openwebtext"

    FILE_PREFIX = DATASET_NAME

    TRAIN_FILENAME = f"{FILE_PREFIX}-train.bin"
    TEST_FILENAME = f"{FILE_PREFIX}-test.bin"

    TEST_SPLIT_PROPORTION = 0.005

    ENCODING = "r50k_base"

    VOCABULARY_SIZE = 50258

    PADDING_INDEX = 50257

    NUM_SHARDS = 1024

    def __init__(
        self,
        root_path: str,
        train: bool = True,
        tokens_per_sample: int = 1024,
        samples_per_epoch: int = 8192,
        num_processes: int = 4,
    ):
        super().__init__()

        if tokens_per_sample < 1:
            raise ValueError(f"Tokens per sample must be greater than 0.")

        if samples_per_epoch < 1:
            raise ValueError(f"Samples per epoch must be greater than 0.")

        train_path = path.join(root_path, self.TRAIN_FILENAME)
        test_path = path.join(root_path, self.TEST_FILENAME)

        self.tokenizer = tiktoken.get_encoding(self.ENCODING)

        self.tokens_per_sample = tokens_per_sample
        self.samples_per_epoch = samples_per_epoch

        self.bin_file_path = path.join(
            root_path, self.TRAIN_FILENAME if train else self.TEST_FILENAME
        )

        if not path.exists(train_path) or not path.exists(test_path):
            self.download_and_preprocess(root_path, num_processes)

    def download_and_preprocess(self, root_path: str, num_processes: int) -> None:
        dataset = load_dataset(self.DATASET_NAME, num_proc=num_processes, split="train")

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

    def tokenize(self, sample: dict) -> dict:
        tokens = self.tokenizer.encode_ordinary(sample["text"])

        tokens.append(self.tokenizer.eot_token)

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

            yield x, y


class Alpaca(Dataset):
    DATASET_NAME = "tatsu-lab/alpaca"

    ENCODING = "r50k_base"

    VOCABULARY_SIZE = 50258

    PADDING_INDEX = 50257

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

    def __getitem__(self, index: int):
        sample = self.dataset[index]

        tokens = self.tokenizer.encode_ordinary(sample["text"])

        tokens.append(self.tokenizer.eot_token)

        end = min(len(tokens), self.max_tokens_per_sample)

        x = torch.tensor(tokens[0 : end - 1], dtype=torch.int64)
        y = torch.tensor(tokens[1:end], dtype=torch.int64)

        delta = self.max_tokens_per_sample - len(tokens)

        x = pad(x, (0, delta), "constant", self.PADDING_INDEX)
        y = pad(y, (0, delta), "constant", self.PADDING_INDEX)

        return x, y

    def __len__(self):
        return len(self.dataset)
