import warnings

import torch

from torch import Tensor

from tiktoken import Encoding


class ChatMLTokenizer:
    """Tokenizer for multi-turn ChatML-formatted messages."""

    CHATML_TEMPLATE = "<|im_start|>{role}\n{message}\n<|im_end|>\n"

    RESPONSE_HEADER = "<|im_start|>assistant\n"

    IGNORE_INDEX = -100

    def __init__(self, tokenizer: Encoding, max_tokens_per_sample: int):
        if max_tokens_per_sample < 1:
            raise ValueError(
                f"Max tokens per sample must be greater than 0, {max_tokens_per_sample} given."
            )

        response_header_tokens = tokenizer.encode(
            self.RESPONSE_HEADER, allowed_special="all"
        )

        response_header_length = len(response_header_tokens)

        self.tokenizer = tokenizer
        self.max_tokens_per_sample = max_tokens_per_sample
        self.response_header_length = response_header_length

    def tokenize_messages(self, messages: list[dict]) -> tuple[Tensor, Tensor]:
        sample, labels = [], []

        total_tokens, has_completion = 0, False

        for message in messages:
            is_completion = message["role"] == "assistant"

            text = self.CHATML_TEMPLATE.format(
                role=message["role"],
                message=message["content"],
            )

            tokens = self.tokenizer.encode(text, allowed_special="all")

            total_tokens += len(tokens)

            sample.extend(tokens)

            if is_completion:
                labels.extend([self.IGNORE_INDEX] * self.response_header_length)

                labels.extend(tokens[self.response_header_length :])

                has_completion = True
            else:
                labels.extend([self.IGNORE_INDEX] * len(tokens))

            if total_tokens > self.max_tokens_per_sample:
                break

        if not has_completion:
            warnings.warn(
                "No completion found in sample, training may be unstable. "
                "Increase max_tokens_per_sample to avoid this warning."
            )

        sample = sample[: self.max_tokens_per_sample + 1]
        labels = labels[: self.max_tokens_per_sample + 1]

        sample = sample[:-1]
        labels = labels[1:]

        x = torch.tensor(sample, dtype=torch.int64)
        y = torch.tensor(labels, dtype=torch.int64)

        assert x.shape == y.shape, "Sample / label shape mismatch"

        return x, y
