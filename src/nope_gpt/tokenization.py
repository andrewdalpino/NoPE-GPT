from tiktoken import Encoding


class ChatMLTokenizer:
    """Tokenizer for multi-turn ChatML-formatted messages."""

    START_TOKEN = "<|im_start|>"
    END_TOKEN = "<|im_end|>"

    CHATML_TEMPLATE = "{START_TOKEN}{role}\n{message}\n{END_TOKEN}\n"

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

        self.tokenizer = tokenizer

    @property
    def vocabulary_size(self) -> int:
        """Return the size of the vocabulary."""

        return self.tokenizer.n_vocab

    def tokenize_message(self, message: dict[str, str]) -> list[int]:
        """Tokenize a single message dict."""

        text = self.CHATML_TEMPLATE.format(
            role=message["role"],
            message=message["content"],
        )

        tokens = self.tokenizer.encode(text, allowed_special="all")

        return tokens
