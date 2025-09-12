from tiktoken import Encoding


class ChatMLTokenizer:
    """Tokenizer for multi-turn ChatML-formatted messages."""

    START_TOKEN = "<|im_start|>"
    END_TOKEN = "<|im_end|>"

    CHATML_TEMPLATE = "<|im_start|>{role}\n{message}\n<|im_end|>\n"

    RESPONSE_HEADER = "<|im_start|>assistant\n"

    def __init__(self, tokenizer: Encoding):
        im_start_index = tokenizer.n_vocab
        im_end_index = im_start_index + 1

        tokenizer = Encoding(
            name=tokenizer.name,
            pat_str=tokenizer._pat_str,
            mergeable_ranks=tokenizer._mergeable_ranks,
            special_tokens={
                **tokenizer._special_tokens,
                self.START_TOKEN: im_start_index,
                self.END_TOKEN: im_end_index,
            },
        )

        response_tokens = tokenizer.encode(self.RESPONSE_HEADER, allowed_special="all")

        self.im_end_index = im_end_index
        self.tokenizer = tokenizer
        self.response_tokens = response_tokens

    @property
    def vocabulary_size(self) -> int:
        return self.tokenizer.n_vocab

    @property
    def stop_tokens(self) -> set[int]:
        return {self.tokenizer.eot_token, self.im_end_index}

    def decode_single_token(self, token: int) -> str:
        """Decode a single token into a string."""

        out = self.tokenizer.decode_single_token_bytes(token).decode(
            "utf-8", errors="replace"
        )

        return out

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

        tokens = self.tokenizer.encode(text, allowed_special="all")

        return tokens
