from typing import Self, Union, Optional
from functools import cached_property
from pathlib import Path
from json import dump as save_json, load as load_json

from tiktoken import Encoding, get_encoding

from huggingface_hub import ModelHubMixin, hf_hub_download


class BaseTokenizer(ModelHubMixin):
    @classmethod
    def from_tiktoken(cls, name: str) -> Self:
        """Instantiate a tokenizer from a pretrained tiktoken tokenizer."""

        return cls(get_encoding(name))

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Union[str, bool, None],
    ):
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="tokenizer.json",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
        )

        with open(config_path, "r") as file:
            config = load_json(file)

        tokenizer = cls.from_tiktoken(config["name"])

        return tokenizer

    def __init__(self, tokenizer: Encoding):
        self.tokenizer = tokenizer

    @property
    def name(self) -> str:
        return self.tokenizer.name

    @property
    def vocabulary_size(self) -> int:
        return self.tokenizer.n_vocab

    @property
    def pad_token(self) -> int:
        return self.tokenizer.eot_token

    @property
    def stop_tokens(self) -> set[int]:
        return {self.tokenizer.eot_token}

    def _save_pretrained(self, save_directory: Union[str, Path]) -> None:
        save_path = save_directory / "tokenizer.json"

        config = {
            "name": self.name,
        }

        with open(save_path, "w") as file:
            save_json(config, file)

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

    def decode_tokens(self, tokens: list[int]) -> str:
        """Decode a list of tokens into text."""

        text = self.tokenizer.decode(tokens).strip()

        return text

    def decode_single_token(self, token: int) -> str:
        """Decode a single token into text."""

        text = self.tokenizer.decode_single_token_bytes(token).decode(
            "utf-8", errors="replace"
        )

        return text


class ChatTokenizer(ModelHubMixin):
    """Tokenizer for multi-turn ChatML-formatted messages with tool calls."""

    IM_START_TOKEN = "<|im_start|>"
    IM_END_TOKEN = "<|im_end|>"

    TOOL_START_INDEX = "<tool_call>"
    TOOL_END_INDEX = "</tool_call>"

    CHATML_TEMPLATE = "<|im_start|>{role}\n{message}\n<|im_end|>\n"

    RESPONSE_HEADER = "<|im_start|>assistant\n"

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Union[str, bool, None],
    ):
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="tokenizer.json",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
        )

        with open(config_path, "r") as file:
            config = load_json(file)

        tokenizer = BaseTokenizer.from_tiktoken(config["name"])

        tokenizer = cls(tokenizer)

        return tokenizer

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

    @property
    def name(self) -> str:
        return self.tokenizer.name

    @property
    def vocabulary_size(self) -> int:
        return self.tokenizer.vocabulary_size

    @property
    def pad_token(self) -> int:
        return self.tokenizer.pad_token

    @cached_property
    def stop_tokens(self) -> set[int]:
        return self.tokenizer.stop_tokens | {self.im_end_index}

    def _save_pretrained(self, save_directory: Union[str, Path]) -> None:
        save_path = save_directory / "tokenizer.json"

        config = {
            "name": self.name,
        }

        with open(save_path, "w") as file:
            save_json(config, file)

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

    def decode_tokens(self, tokens: list[int]) -> str:
        return self.tokenizer.decode_tokens(tokens)

    def decode_single_token(self, token: int) -> str:
        return self.tokenizer.decode_single_token(token)
