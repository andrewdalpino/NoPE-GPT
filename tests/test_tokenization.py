import unittest

from unittest.mock import patch, mock_open

from pathlib import Path

from src.nope_gpt.tokenization import BaseTokenizer, ChatTokenizer


class MockEncoding:
    def __init__(
        self,
        name="mock_encoding",
        pat_str="",
        mergeable_ranks=None,
        special_tokens=None,
        eot_token=1,
    ):
        self.name = name
        self._pat_str = pat_str
        self._mergeable_ranks = mergeable_ranks or {}
        self._special_tokens = special_tokens or {}
        self.eot_token = eot_token
        self.n_vocab = 1000

    def encode_ordinary(self, text):
        # For testing, return simple predictable tokens
        return [ord(c) % 1000 for c in text]

    def encode(self, text, allowed_special=None):
        # For testing, return simple predictable tokens
        if allowed_special == "all":
            # Simulate a special token at the beginning
            return [500] + [ord(c) % 1000 for c in text]
        return [ord(c) % 1000 for c in text]

    def decode(self, tokens):
        # Simple decoding for testing
        return "".join([chr(t) for t in tokens])

    def decode_single_token_bytes(self, token):
        # Return bytes version of the token for testing
        return str(token).encode("utf-8")


class TestBaseTokenizer(unittest.TestCase):
    def setUp(self):
        """Set up test parameters."""
        self.mock_encoding = MockEncoding()
        self.tokenizer = BaseTokenizer(self.mock_encoding)

    @patch("src.nope_gpt.tokenization.get_encoding")
    def test_from_tiktoken(self, mock_get_encoding):
        mock_get_encoding.return_value = self.mock_encoding

        tokenizer = BaseTokenizer.from_tiktoken("test_encoding")

        mock_get_encoding.assert_called_once_with("test_encoding")
        self.assertIsInstance(tokenizer, BaseTokenizer)
        self.assertEqual(tokenizer.tokenizer, self.mock_encoding)

    @patch("src.nope_gpt.tokenization.hf_hub_download")
    @patch("src.nope_gpt.tokenization.load_json")
    @patch("builtins.open", new_callable=mock_open)
    def test_from_pretrained(self, mock_file, mock_load_json, mock_hf_hub_download):
        mock_hf_hub_download.return_value = "fake/path/tokenizer.json"
        mock_load_json.return_value = {"name": "test_encoding"}

        # Need to patch the from_tiktoken method as well
        with patch.object(
            BaseTokenizer, "from_tiktoken", return_value=self.tokenizer
        ) as mock_from_tiktoken:
            tokenizer = BaseTokenizer._from_pretrained(
                model_id="test/model",
                revision=None,
                cache_dir=None,
                force_download=False,
                proxies=None,
                resume_download=None,
                local_files_only=False,
                token=None,
            )

            mock_hf_hub_download.assert_called_once()
            mock_file.assert_called_once_with("fake/path/tokenizer.json", "r")
            mock_load_json.assert_called_once()
            mock_from_tiktoken.assert_called_once_with("test_encoding")

            self.assertEqual(tokenizer, self.tokenizer)

    def test_properties(self):
        self.assertEqual(self.tokenizer.name, "mock_encoding")
        self.assertEqual(self.tokenizer.vocabulary_size, 1000)
        self.assertEqual(self.tokenizer.pad_token, 1)
        self.assertEqual(self.tokenizer.stop_tokens, {1})

    @patch("src.nope_gpt.tokenization.save_json")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_pretrained(self, mock_file, mock_save_json):
        save_dir = Path("/fake/save/dir")

        self.tokenizer._save_pretrained(save_dir)

        expected_save_path = save_dir / "tokenizer.json"
        mock_file.assert_called_once_with(expected_save_path, "w")
        mock_save_json.assert_called_once_with({"name": "mock_encoding"}, mock_file())

    @patch("src.nope_gpt.tokenization.Encoding")
    def test_add_special_tokens(self, mock_encoding_class):
        # Create a new mock encoding that will be returned by the Encoding constructor
        new_mock_encoding = MockEncoding(
            name="mock_encoding",
            special_tokens={
                **self.mock_encoding._special_tokens,
                "special1": 1000,
                "special2": 1001,
            },
        )
        mock_encoding_class.return_value = new_mock_encoding

        self.tokenizer.add_special_tokens(["special1", "special2"])

        # Check that Encoding was called with the correct arguments
        mock_encoding_class.assert_called_once()

        # Check that the tokenizer was updated
        self.assertEqual(self.tokenizer.tokenizer, new_mock_encoding)

    def test_tokenize(self):
        tokens = self.tokenizer.tokenize("test")
        # Our mock encoding returns ord(c) % 1000 for each character
        expected = [116, 101, 115, 116]  # ASCII values for 'test'
        self.assertEqual(tokens, expected)

    def test_tokenize_with_special(self):
        tokens = self.tokenizer.tokenize_with_special("test")
        # Our mock adds a 500 at the beginning for special tokens
        expected = [500, 116, 101, 115, 116]
        self.assertEqual(tokens, expected)

    def test_tokenize_single(self):
        # Mock the tokenize_with_special to return a single token
        with patch.object(self.tokenizer, "tokenize_with_special", return_value=[42]):
            token = self.tokenizer.tokenize_single("a")
            self.assertEqual(token, 42)

        # Test the error case
        with patch.object(self.tokenizer, "tokenize_with_special", return_value=[1, 2]):
            with self.assertRaises(ValueError):
                self.tokenizer.tokenize_single("ab")

    def test_decode_tokens(self):
        # Mock decode to return predictable output
        self.mock_encoding.decode = lambda tokens: "".join([chr(t) for t in tokens])

        text = self.tokenizer.decode_tokens([97, 98, 99])  # ASCII for 'abc'
        self.assertEqual(text, "abc")

    def test_decode_single_token(self):
        text = self.tokenizer.decode_single_token(42)
        self.assertEqual(text, "42")


class TestChatTokenizer(unittest.TestCase):
    """Test cases for the ChatTokenizer class."""

    def setUp(self):
        # Create a BaseTokenizer with a mock encoding
        self.mock_encoding = MockEncoding()
        self.base_tokenizer = BaseTokenizer(self.mock_encoding)

        # Patch the add_special_tokens and tokenize_single methods
        with (
            patch.object(self.base_tokenizer, "add_special_tokens"),
            patch.object(self.base_tokenizer, "tokenize_single", return_value=999),
            patch.object(
                self.base_tokenizer, "tokenize_with_special", return_value=[1, 2, 3]
            ),
        ):

            self.chat_tokenizer = ChatTokenizer(self.base_tokenizer)

    @patch("src.nope_gpt.tokenization.hf_hub_download")
    @patch("src.nope_gpt.tokenization.load_json")
    @patch("builtins.open", new_callable=mock_open)
    def test_from_pretrained(self, mock_file, mock_load_json, mock_hf_hub_download):

        mock_hf_hub_download.return_value = "fake/path/tokenizer.json"
        mock_load_json.return_value = {"name": "test_encoding"}

        # Need to patch several methods
        with (
            patch.object(
                BaseTokenizer, "from_tiktoken", return_value=self.base_tokenizer
            ),
            patch.object(ChatTokenizer, "__init__", return_value=None) as mock_init,
        ):

            tokenizer = ChatTokenizer._from_pretrained(
                model_id="test/model",
                revision=None,
                cache_dir=None,
                force_download=False,
                proxies=None,
                resume_download=None,
                local_files_only=False,
                token=None,
            )

            mock_hf_hub_download.assert_called_once()
            mock_file.assert_called_once_with("fake/path/tokenizer.json", "r")
            mock_load_json.assert_called_once()
            mock_init.assert_called_once()

    def test_init(self):
        # Check that add_special_tokens was called with the correct tokens
        with (
            patch.object(self.base_tokenizer, "add_special_tokens") as mock_add_special,
            patch.object(self.base_tokenizer, "tokenize_single", return_value=999),
            patch.object(
                self.base_tokenizer, "tokenize_with_special", return_value=[1, 2, 3]
            ),
        ):

            ChatTokenizer(self.base_tokenizer)

            expected_tokens = [
                ChatTokenizer.IM_START_TOKEN,
                ChatTokenizer.IM_END_TOKEN,
                ChatTokenizer.TOOL_START_INDEX,
                ChatTokenizer.TOOL_END_INDEX,
            ]
            mock_add_special.assert_called_once_with(expected_tokens)

    def test_properties(self):
        self.assertEqual(self.chat_tokenizer.name, "mock_encoding")
        self.assertEqual(self.chat_tokenizer.vocabulary_size, 1000)
        self.assertEqual(self.chat_tokenizer.pad_token, 1)

        # stop_tokens is a cached_property that combines base tokenizer stop tokens with im_end_index
        self.assertEqual(self.chat_tokenizer.stop_tokens, {1, 999})

    @patch("src.nope_gpt.tokenization.save_json")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_pretrained(self, mock_file, mock_save_json):
        """Test _save_pretrained method."""
        save_dir = Path("/fake/save/dir")

        self.chat_tokenizer._save_pretrained(save_dir)

        expected_save_path = save_dir / "tokenizer.json"
        mock_file.assert_called_once_with(expected_save_path, "w")
        mock_save_json.assert_called_once_with({"name": "mock_encoding"}, mock_file())

    def test_tokenize_message(self):
        # Patch tokenize_with_special to return predictable tokens
        with patch.object(
            self.base_tokenizer, "tokenize_with_special", return_value=[101, 102, 103]
        ):
            tokens = self.chat_tokenizer.tokenize_message(
                {"role": "user", "content": "Hello"}
            )

            self.assertEqual(tokens, [101, 102, 103])

            # Check the formatting
            expected_text = ChatTokenizer.CHATML_TEMPLATE.format(
                role="user", message="Hello"
            )
            self.base_tokenizer.tokenize_with_special.assert_called_once_with(
                expected_text
            )

    def test_tokenize_prompt(self):
        # Create a list of messages
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        # Patch tokenize_message to return predictable tokens
        with patch.object(
            self.chat_tokenizer, "tokenize_message"
        ) as mock_tokenize_message:
            mock_tokenize_message.side_effect = [[101, 102], [201, 202]]

            tokens = self.chat_tokenizer.tokenize_prompt(messages)

            # Should contain tokens from both messages plus response_tokens
            expected = [
                101,
                102,
                201,
                202,
                1,
                2,
                3,
            ]  # response_tokens was set to [1, 2, 3] in setUp
            self.assertEqual(tokens, expected)

            # Check that tokenize_message was called for each message
            self.assertEqual(mock_tokenize_message.call_count, 2)
            mock_tokenize_message.assert_any_call(messages[0])
            mock_tokenize_message.assert_any_call(messages[1])

    def test_decode_methods(self):
        # These methods just delegate to the base tokenizer, so check that they do
        with (
            patch.object(self.base_tokenizer, "decode_tokens") as mock_decode_tokens,
            patch.object(
                self.base_tokenizer, "decode_single_token"
            ) as mock_decode_single,
        ):

            self.chat_tokenizer.decode_tokens([1, 2, 3])
            self.chat_tokenizer.decode_single_token(42)

            mock_decode_tokens.assert_called_once_with([1, 2, 3])
            mock_decode_single.assert_called_once_with(42)


if __name__ == "__main__":
    unittest.main()
