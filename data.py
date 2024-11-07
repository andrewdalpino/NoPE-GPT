import random
from os import path

from datasets import load_dataset

import numpy as np

import torch

import tiktoken

from torch.utils.data import IterableDataset

from typing import Dict

from tqdm import tqdm


class OpenwebtextDataset(IterableDataset):
    DATASET_NAME = "openwebtext"

    TRAIN_FILENAME = "train.bin"
    TEST_FILENAME = "test.bin"

    TEST_SPLIT_PROPORTION = 0.0015

    ENCODING = "r50k_base"
    VOCABULARY_SIZE = 50257

    NUM_SHARDS = 1024

    def __init__(
        self,
        root_path: str,
        train: bool = True,
        block_size: int = 1024,
        max_samples_per_epoch: int = 8192,
        num_processes: int = 8,
    ):
        if block_size < 1:
            raise ValueError(f"Block size must be greater than 0, {block_size} given.")

        if max_samples_per_epoch < 1:
            raise ValueError(f"Max samples per epoch must be greater than 0.")

        if num_processes < 1:
            raise ValueError(
                f"Num processes must be greater than 0, {num_processes} given."
            )

        train_path = path.join(root_path, self.TRAIN_FILENAME)
        test_path = path.join(root_path, self.TEST_FILENAME)

        self.tokenizer = tiktoken.get_encoding(self.ENCODING)

        self.bin_file_path = path.join(
            root_path, self.TRAIN_FILENAME if train else self.TEST_FILENAME
        )

        if not path.exists(train_path) or not path.exists(test_path):
            self.download_and_preprocess(root_path, num_processes)

        self.block_size = block_size
        self.max_samples_per_epoch = max_samples_per_epoch

    def download_and_preprocess(self, root_path: str, num_processes: int) -> None:
        dataset = load_dataset(self.DATASET_NAME, num_proc=num_processes)

        splits = dataset["train"].train_test_split(
            test_size=self.TEST_SPLIT_PROPORTION, shuffle=True
        )

        splits = splits.map(
            self.tokenize,
            desc="Tokenizing",
            remove_columns=["text"],
            num_proc=num_processes,
        )

        for split, dataset in splits.items():
            filename = path.join(root_path, f"{split}.bin")

            length = np.sum(dataset["length"], dtype=np.uint64)

            bin_out = np.memmap(filename, dtype=np.uint16, mode="w+", shape=length)

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

    def tokenize(self, sample: Dict) -> Dict:
        tokens = self.tokenizer.encode_ordinary(sample["text"])

        tokens.append(self.tokenizer.eot_token)

        return {
            "tokens": tokens,
            "length": len(tokens),
        }

    def __iter__(self):
        data = np.memmap(self.bin_file_path, dtype=np.uint16, mode="r")

        max_start = len(data) - (self.block_size + 1)

        num_samples = min(self.max_samples_per_epoch, max_start)

        for i in range(num_samples):
            start = random.randint(0, max_start)

            x = data[start : start + self.block_size]
            y = data[start + 1 : start + 1 + self.block_size]

            x = torch.from_numpy(x.astype(np.int64))
            y = torch.from_numpy(y.astype(np.int64))

            yield x, y
