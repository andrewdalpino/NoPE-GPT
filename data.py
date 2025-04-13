import random
import warnings

from os import path, remove as delete_file
from functools import partial
from copy import deepcopy

from datasets import load_dataset

from tiktoken import Encoding

import numpy as np

import torch

from torch import Tensor
from torch.utils.data import IterableDataset, Dataset
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

CHATML_TEMPLATE = "<|im_start|>{role}\n{message}\n<|im_end|>\n"

RESPONSE_HEADER = "<|im_start|>assistant\n"

IGNORE_INDEX = -100


class Fineweb(IterableDataset):
    """
    The Fineweb dataset by HuggingFace, tokenized, and stored as a memory map for fast access
    during pre-training. Endlessly emits random chunks of tokens from the corpus in infinitum.
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


class ChatMLTokenizer:
    """Tokenizer for multi-turn ChatML-formatted messages."""

    def __init__(
        self,
        tokenizer: Encoding,
        max_tokens_per_sample: int = 1024,
        train_on_inputs: bool = False,
    ):
        if max_tokens_per_sample < 1:
            raise ValueError(
                f"Max tokens per sample must be greater than 0, {max_tokens_per_sample} given."
            )

        response_header_tokens = tokenizer.encode(
            RESPONSE_HEADER, allowed_special="all"
        )
        
        response_header_length = len(response_header_tokens)

        self.tokenizer = tokenizer
        self.max_tokens_per_sample = max_tokens_per_sample
        self.train_on_inputs = train_on_inputs
        self.response_header_length = response_header_length

    def tokenize_messages(self, messages: list[dict]) -> tuple[Tensor, Tensor]:
        sample, labels = [], []

        total_tokens, has_completion = 0, False

        for message in messages:
            is_completion = message["role"] == "assistant"

            text = CHATML_TEMPLATE.format(
                role=message["role"],
                message=message["content"],
            )

            tokens = self.tokenizer.encode(text, allowed_special="all")

            total_tokens += len(tokens)

            sample.extend(tokens)

            if self.train_on_inputs:
                labels.extend(tokens)

                continue

            if is_completion:
                labels.extend([IGNORE_INDEX] * self.response_header_length)

                labels.extend(tokens[self.response_header_length :])

                has_completion = True
            else:
                labels.extend([IGNORE_INDEX] * len(tokens))

            if total_tokens > self.max_tokens_per_sample:
                break

        if not self.train_on_inputs and not has_completion:
            warnings.warn(
                "No completion found in sample, training may be unstable. "
                "Increase max_tokens_per_sample or train on inputs."
            )

        sample = sample[: self.max_tokens_per_sample + 1]
        labels = labels[: self.max_tokens_per_sample + 1]

        sample = sample[:-1]
        labels = labels[1:]

        x = torch.tensor(sample, dtype=torch.int64)
        y = torch.tensor(labels, dtype=torch.int64)

        assert x.shape == y.shape, "Sample / label shape mismatch"

        return x, y


class SmolTalk(Dataset):
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
        subset: str = "all",
    ):
        super().__init__()

        if subset not in self.SUBSETS:
            raise ValueError(f"Invalid subset, {subset} given.")

        self.tokenizer = tokenizer

        self.dataset = load_dataset(self.DATASET_NAME, subset, split="train")

    def __getitem__(self, index: int):
        messages = self.dataset[index]["messages"]

        x, y = self.tokenizer.tokenize_messages(messages)

        return x, y

    def __len__(self):
        return len(self.dataset)


class UltraFeedbackSFT(Dataset):
    """
    The binarized version of the UltraFeedback dataset for human preference alignment via SFT.
    """

    DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"

    def __init__(
        self,
        tokenizer: ChatMLTokenizer,
        split: str = "train",
    ):
        super().__init__()

        if split not in {"train", "test"}:
            raise ValueError(f"Split must be either train or test, {split} given.")

        self.tokenizer = tokenizer

        new_dataset = partial(load_dataset, name=self.DATASET_NAME)

        if split == "train":
            self.dataset = new_dataset(split="train_sft")
        else:
            self.dataset = new_dataset(split="test_sft")

    def __getitem__(self, index: int):
        messages = self.dataset[index]["messages"]

        x, y = self.tokenizer.tokenize_messages(messages)

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
