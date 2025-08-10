import warnings

from os import path, remove as delete_file
from functools import partial

from datasets import load_dataset

from tiktoken import Encoding

import numpy as np

import torch

from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.nope_gpt.tokenization import ChatMLTokenizer

from tqdm import tqdm

IGNORE_INDEX = -100


class Fineweb(Dataset):
    """
    The Fineweb dataset by HuggingFace, tokenized, and stored as a memory map for fast access
    during pre-training.
    """

    DATASET_NAME = "HuggingFaceFW/fineweb"

    SUBSETS = frozenset({"sample-10BT", "sample-100BT", "sample-350BT"})

    def __init__(
        self,
        tokenizer: Encoding,
        root_path: str,
        subset: str | None,
        tokens_per_sample: int,
    ):
        super().__init__()

        assert subset is None or subset in self.SUBSETS, (
            f"Invalid subset, {subset} given. " f"Valid subsets are: {self.SUBSETS}"
        )

        assert (
            tokens_per_sample > 0
        ), f"Tokens per sample must be greater than 0, {tokens_per_sample} given."

        dataset_name = f"fineweb-{subset}" if subset is not None else "fineweb"

        bin_path = path.join(root_path, f"{dataset_name}-{tokenizer.name}.bin")

        self.tokenizer = tokenizer

        if not path.exists(bin_path):
            dataset = load_dataset(
                self.DATASET_NAME,
                name=subset,
                split="train",
                streaming=True,
            )

            tokenized_dataset = dataset.map(
                self.tokenize,
                remove_columns=["text"],
                desc="Tokenizing Fineweb dataset",
            )

            temp_path = bin_path + ".temp"

            temp_out = np.memmap(temp_path, dtype=np.uint16, mode="w+", shape=1024)

            new_memmap = partial(np.memmap, temp_path, dtype=np.uint16, mode="r+")

            total_length = 0

            for row in tqdm(tokenized_dataset, desc=f"Preprocessing dataset"):
                tokens = row["tokens"]

                length = len(tokens)

                l_hat = total_length + length

                while l_hat > len(temp_out):
                    new_temp_out = new_memmap(shape=2 * len(temp_out))

                    new_temp_out[: len(temp_out)] = temp_out

                    temp_out = new_temp_out

                temp_out[total_length:l_hat] = tokens

                total_length = l_hat

            bin_out = np.memmap(
                bin_path, dtype=np.uint16, mode="w+", shape=total_length
            )

            bin_out[:] = temp_out[:total_length]

            bin_out.flush()

            delete_file(temp_path)

        memmap = np.memmap(bin_path, dtype=np.uint16, mode="r")

        max_offset = len(memmap) // tokens_per_sample

        self.memmap = memmap
        self.tokens_per_sample = tokens_per_sample
        self.max_offset = max_offset

    def tokenize(self, sample: dict) -> dict:
        tokens = self.tokenizer.encode_ordinary(sample["text"])

        tokens.append(self.tokenizer.eot_token)

        return {
            "tokens": tokens,
        }

    def __getitem__(self, index: int):
        assert index < self.max_offset, "Offset out of bounds."

        start: int = index * self.tokens_per_sample
        end: int = start + self.tokens_per_sample

        x = self.memmap[start:end]
        y = self.memmap[start + 1 : end + 1]

        x = x.astype(np.int64)
        y = y.astype(np.int64)

        assert x.shape == y.shape, "Sample / label shape mismatch."

        return x, y

    def __len__(self):
        return self.max_offset


class ChatMLDataset(Dataset):
    """Base class for datasets with samples formatted as ChatML messages."""

    IGNORE_INDEX = -100

    def __init__(self, tokenizer: ChatMLTokenizer, max_tokens_per_sample: int):
        assert max_tokens_per_sample > 0, "max_tokens_per_sample must be positive"

        self.tokenizer = tokenizer
        self.max_tokens_per_sample = max_tokens_per_sample

    def tokenize_messages(self, messages: list[dict]) -> tuple[list, list]:
        sample, labels = [], []

        total_tokens = 0
        has_completion = False

        for message in messages:
            tokens = self.tokenizer.tokenize_message(message)

            total_tokens += len(tokens)

            if total_tokens > 1 + self.max_tokens_per_sample:
                break

            sample.extend(tokens)

            is_completion = message["role"] == "assistant"

            if is_completion:
                labels.extend(tokens)

                has_completion = True
            else:
                labels.extend([self.IGNORE_INDEX] * len(tokens))

        if not has_completion:
            warnings.warn(
                "No completion found in sample, training may be unstable. "
                "Filter or increase max_tokens_per_sample to avoid this warning."
            )

        sample = sample[:-1]
        labels = labels[1:]

        return sample, labels


class SmolTalk(ChatMLDataset):
    """
    The SmolTalk dataset by HuggingFace formatted as ChatML messages for supervised instruction tuning.
    """

    DATASET_NAME = "HuggingFaceTB/smoltalk"

    SUBSETS = frozenset(
        {
            "all",
            "apigen-80k",
            "everyday-conversations",
            "explore-instruct-rewriting",
            "longalign",
            "metamathqa-50k",
            "numina-cot-100k",
            "openhermes-100k",
            "self-oss-instruct",
            "smol-constraints",
            "smol-magpie-ultra",
            "smol-rewrite",
            "smol-summarize",
            "systemchats-30k",
        }
    )

    def __init__(
        self,
        tokenizer: ChatMLTokenizer,
        subset: str,
        max_tokens_per_sample: int,
        filter_long_samples: bool,
    ):
        super().__init__(tokenizer, max_tokens_per_sample)

        assert (
            subset in self.SUBSETS
        ), f"Invalid subset, {subset} given, valid subsets are: {self.SUBSETS}"

        dataset = load_dataset(self.DATASET_NAME, subset, split="train")

        def filter_by_max_tokens(sample: dict) -> bool:
            tokens = []

            for message in sample["messages"]:
                tokens.extend(tokenizer.tokenize_message(message))

            keep = len(tokens) < max_tokens_per_sample

            return keep

        if filter_long_samples:
            dataset = dataset.filter(
                filter_by_max_tokens,
                desc=f"Filtering samples longer than {max_tokens_per_sample} tokens",
            )

        self.dataset = dataset

    def __getitem__(self, index: int):
        messages = self.dataset[index]["messages"]

        sample, labels = self.tokenize_messages(messages)

        x = torch.tensor(sample, dtype=torch.int64)
        y = torch.tensor(labels, dtype=torch.int64)

        assert x.shape == y.shape, "Sample / label shape mismatch."

        return x, y

    def __len__(self):
        return len(self.dataset)


class UltraFeedbackSFT(ChatMLDataset):
    """
    The binarized version of the UltraFeedback dataset for human preference alignment via SFT.
    """

    DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"

    def __init__(
        self,
        tokenizer: ChatMLTokenizer,
        split: str,
        max_tokens_per_sample: int,
        filter_long_samples: bool,
    ):
        super().__init__(tokenizer, max_tokens_per_sample)

        assert split in {"train", "test"}, (
            f"Invalid split, {split} given. " f"Valid splits are: 'train', 'test'."
        )

        new_dataset = partial(load_dataset, name=self.DATASET_NAME)

        if split == "train":
            dataset = new_dataset(split="train_sft")
        else:
            dataset = new_dataset(split="test_sft")

        def filter_by_max_tokens(sample: dict) -> bool:
            tokens = []

            for message in sample["messages"]:
                tokens.extend(tokenizer.tokenize_message(message))

            keep = len(tokens) < max_tokens_per_sample

            return keep

        if filter_long_samples:
            dataset = dataset.filter(
                filter_by_max_tokens,
                desc=f"Filtering samples longer than {max_tokens_per_sample} tokens",
            )

        self.dataset = dataset

    def __getitem__(self, index: int):
        messages = self.dataset[index]["messages"]

        sample, labels = self.tokenize_messages(messages)

        x = torch.tensor(sample, dtype=torch.int64)
        y = torch.tensor(labels, dtype=torch.int64)

        assert x.shape == y.shape, "Sample / label shape mismatch."

        return x, y

    def __len__(self):
        return len(self.dataset)


def pad_collate(
    batch: list[tuple[Tensor, Tensor]], padding_side: str, padding_index: int
) -> tuple[Tensor, Tensor]:
    """Custom collate function adds padding to batched samples."""

    samples, labels = zip(*batch)

    x = pad_sequence(
        list(samples),
        batch_first=True,
        padding_value=padding_index,
        padding_side=padding_side,
    )

    y = pad_sequence(
        list(labels),
        batch_first=True,
        padding_value=IGNORE_INDEX,
        padding_side=padding_side,
    )

    assert x.shape == y.shape, "Batch shape mismatch"

    return x, y
