from typing import Self
from functools import cached_property

from tiktoken import Encoding, get_encoding


class BaseTokenizer:
    @classmethod
    def from_pretrained(cls, name: str) -> Self:
        """Instantiate a tokenizer from a pretrained tiktoken tokenizer."""

        return cls(get_encoding(name))

    def __init__(self, tokenizer: Encoding):
        self.tokenizer = tokenizer

    @property
    def name(self) -> str:
        return self.tokenizer.name

    @property
    def vocabulary_size(self) -> int:
        return self.tokenizer.n_vocab

    @property
    def eos_token(self) -> int:
        return self.tokenizer.eot_token

    @property
    def stop_tokens(self) -> set[int]:
        return {self.eos_token}

    def add_special_tokens(self, tokens: list[str]) -> None:
        start_index = self.vocabulary_size

        new_tokens = {token: start_index + i for i, token in enumerate(tokens)}

        tokenizer = Encoding(
            name=self.tokenizer.name,
            pat_str=self.tokenizer._pat_str,
            mergeable_ranks=self.tokenizer._mergeable_ranks,
            special_tokens={
                **self.tokenizer._special_tokens,
                **new_tokens,
            },
        )

        self.tokenizer = tokenizer

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text without special tokens."""

        return self.tokenizer.encode_ordinary(text)

    def tokenize_with_special(self, text: str) -> list[int]:
        """Tokenize text and include special tokens."""

        return self.tokenizer.encode(text, allowed_special="all")

    def tokenize_single(self, text: str) -> int:
        """
        Tokenize a single token and return it. If the text does not correspond to a
        single token, an error is raised.
        """

        tokens = self.tokenize_with_special(text)

        if len(tokens) != 1:
            raise ValueError(f"Input text '{text}' is not a single token.")

        return tokens[0]

    def decode_single_token(self, token: int) -> str:
        """Decode a single token into text."""

        text = self.tokenizer.decode_single_token_bytes(token).decode(
            "utf-8", errors="replace"
        )

        return text


class ChatTokenizer:
    """Tokenizer for multi-turn ChatML-formatted messages with tool calls."""

    IM_START_TOKEN = "<|im_start|>"
    IM_END_TOKEN = "<|im_end|>"

    TOOL_START_INDEX = "<tool_call>"
    TOOL_END_INDEX = "</tool_call>"

    CHATML_TEMPLATE = "<|im_start|>{role}\n{message}\n<|im_end|>\n"

    RESPONSE_HEADER = "<|im_start|>assistant\n"

    def __init__(self, tokenizer: BaseTokenizer):
        tokenizer.add_special_tokens(
            [
                self.IM_START_TOKEN,
                self.IM_END_TOKEN,
                self.TOOL_START_INDEX,
                self.TOOL_END_INDEX,
            ]
        )

        im_end_index = tokenizer.tokenize_single(self.IM_END_TOKEN)

        response_tokens = tokenizer.tokenize_with_special(self.RESPONSE_HEADER)

        self.im_end_index = im_end_index
        self.response_tokens = response_tokens
        self.tokenizer = tokenizer

    @cached_property
    def stop_tokens(self) -> set[int]:
        return self.tokenizer.stop_tokens | {self.im_end_index}

    def tokenize_prompt(self, messages: list[dict]) -> list[int]:
        """Tokenize a list of messages and add a response header."""

        tokens = []

        for message in messages:
            tokens.extend(self.tokenize_message(message))

        tokens.extend(self.response_tokens)

        return tokens

    def tokenize_message(self, message: dict[str, str]) -> list[int]:
        """Tokenize a single message dict."""

        text = self.CHATML_TEMPLATE.format(
            role=message["role"],
            message=message["content"],
        )

        tokens = self.tokenizer.tokenize_with_special(text)

        return tokens

    def decode_single_token(self, token: int) -> str:
        return self.tokenizer.decode_single_token(token)
