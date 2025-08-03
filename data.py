from os import path, remove as delete_file
from functools import partial

from datasets import load_dataset

from tiktoken import Encoding

import numpy as np

from numpy import ndarray

from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from tokenization import ChatMLTokenizer

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

        if subset is not None:
            if subset not in self.SUBSETS:
                raise ValueError(f"Invalid subset, {subset} given.")

        if tokens_per_sample < 1:
            raise ValueError(f"Tokens per sample must be greater than 0.")

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
        assert index < self.max_offset, "Offset out of bounds"

        start: int = index * self.tokens_per_sample
        end: int = start + self.tokens_per_sample

        x = self.memmap[start:end]
        y = self.memmap[start + 1 : end + 1]

        x = x.astype(np.int64)
        y = y.astype(np.int64)

        assert x.shape == y.shape, "Sample / label shape mismatch"

        return x, y

    def __len__(self):
        return self.max_offset


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
