import random

from os import path, remove as delete_file
from functools import partial

from datasets import load_dataset

from tiktoken import Encoding

import numpy as np

import torch

from torch import Tensor
from torch.utils.data import IterableDataset, Dataset
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm


CHATML_TEMPLATE = "<|im_start|>{role}\n{message}\n<|im_end|>\n"

PADDING_INDEX = 0
IGNORE_INDEX = -100


class Fineweb(IterableDataset):
    """
    The Fineweb dataset by HuggingFace, tokenized and stored as a memory map for fast
    access during pre-training.
    """

    DATASET_NAME = "HuggingFaceFW/fineweb"

    SUBSETS = frozenset({"sample-10BT", "sample-100BT", "sample-350BT"})

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

        if subset is not None:
            if subset not in self.SUBSETS:
                raise ValueError(f"Invalid subset, {subset} given.")

        if split not in {"train", "test"}:
            raise ValueError(f"Split must be either train or test, {split} given.")

        if tokens_per_sample < 1:
            raise ValueError(f"Tokens per sample must be greater than 0.")

        if samples_per_epoch < 1:
            raise ValueError(f"Samples per epoch must be greater than 0.")

        dataset_name = f"fineweb-{subset}" if subset is not None else "fineweb"

        bin_path = path.join(root_path, f"{dataset_name}-{tokenizer.name}.bin")

        self.tokenizer = tokenizer

        if not path.exists(bin_path):
            tokenized_dataset = load_dataset(
                self.DATASET_NAME,
                name=subset,
                split="train",
                streaming=True,
            ).map(
                self.tokenize,
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

        tokens_per_epoch = samples_per_epoch * (tokens_per_sample + 1)

        start = tokens_per_epoch if split == "train" else 0
        end = len(memmap) if split == "train" else tokens_per_epoch

        max_offset = end - tokens_per_epoch

        self.memmap = memmap
        self.tokens_per_sample = tokens_per_sample
        self.samples_per_epoch = samples_per_epoch
        self.start = start
        self.max_offset = max_offset

    def tokenize(self, sample: dict) -> dict:
        tokens = self.tokenizer.encode_ordinary(sample["text"])

        tokens.append(self.tokenizer.eot_token)

        return {
            "tokens": tokens,
        }

    def __iter__(self):
        start = random.randint(self.start, self.max_offset)

        for _ in range(self.samples_per_epoch):
            end = start + self.tokens_per_sample

            x = self.memmap[start:end]
            y = self.memmap[start + 1 : end + 1]

            x = x.astype(np.int64)
            y = y.astype(np.int64)

            assert x.shape == y.shape, "Sample / label shape mismatch"

            yield x, y

            start += self.tokens_per_sample


class ChatMLMixin:
    """Mixin class for datasets that use the ChatML message format."""

    def __init__(
        self,
        tokenizer: Encoding,
        max_tokens_per_sample: int = 1024,
        train_on_inputs: bool = False,
    ):
        super().__init__()

        if max_tokens_per_sample < 1:
            raise ValueError(
                f"Max tokens per sample must be greater than 0, {max_tokens_per_sample} given."
            )

        self.tokenizer = tokenizer
        self.max_tokens_per_sample = max_tokens_per_sample
        self.train_on_inputs = train_on_inputs

    def tokenize_messages(self, messages: list[dict]) -> tuple[Tensor, Tensor]:
        sample, label = [], []

        for message in messages:
            is_end_of_turn = message["role"] == "assistant"

            text = CHATML_TEMPLATE.format(
                role=message["role"],
                message=message["content"],
            )

            tokens = self.tokenizer.encode(text, allowed_special="all")

            if is_end_of_turn:
                tokens.append(self.tokenizer.eot_token)

            sample.extend(tokens[:-1])

            if is_end_of_turn or self.train_on_inputs:
                label.extend(tokens[1:])
            else:
                label.extend([IGNORE_INDEX] * (len(tokens) - 1))

            if len(sample) >= self.max_tokens_per_sample:
                break

        sample = sample[: self.max_tokens_per_sample]
        label = label[: self.max_tokens_per_sample]

        x = torch.tensor(sample, dtype=torch.int64)
        y = torch.tensor(label, dtype=torch.int64)

        assert x.shape == y.shape, "Sample / label shape mismatch"

        return x, y


class SmolTalk(ChatMLMixin, Dataset):
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
        tokenizer: Encoding,
        subset: str = "all",
        max_tokens_per_sample: int = 1024,
        train_on_inputs: bool = False,
    ):
        super().__init__(tokenizer, max_tokens_per_sample, train_on_inputs)

        if subset not in self.SUBSETS:
            raise ValueError(f"Invalid subset, {subset} given.")

        self.dataset = load_dataset(self.DATASET_NAME, subset, split="train")

    def __getitem__(self, index: int):
        messages = self.dataset[index]["messages"]

        x, y = self.tokenize_messages(messages)

        return x, y

    def __len__(self):
        return len(self.dataset)


class UltraFeedback(ChatMLMixin, Dataset):
    """
    The binarized version of the UltraFeedback dataset for human preference alignment via SFT.
    """

    DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"

    def __init__(
        self,
        tokenizer: Encoding,
        split: str = "train",
        max_tokens_per_sample: int = 1024,
        train_on_inputs: bool = False,
    ):
        super().__init__(tokenizer, max_tokens_per_sample, train_on_inputs)

        if split not in {"train", "test"}:
            raise ValueError(f"Split must be either train or test, {split} given.")

        new_dataset = partial(load_dataset, name=self.DATASET_NAME)

        if split == "train":
            self.dataset = new_dataset(split="train_sft")
        else:
            self.dataset = new_dataset(split="test_sft")

    def __getitem__(self, index: int):
        messages = self.dataset[index]["messages"]

        x, y = self.tokenize_messages(messages)

        return x, y

    def __len__(self):
        return len(self.dataset)


def left_pad_collate(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """Custom collate function adds left padding to batched samples."""

    samples, labels = zip(*batch)

    x = pad_sequence(
        list(samples),
        batch_first=True,
        padding_value=PADDING_INDEX,
        padding_side="left",
    )

    y = pad_sequence(
        list(labels),
        batch_first=True,
        padding_value=IGNORE_INDEX,
        padding_side="left",
    )

    assert x.shape == y.shape, "Batch shape mismatch"

    return x, y
