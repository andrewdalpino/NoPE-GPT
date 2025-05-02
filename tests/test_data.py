import unittest
import torch
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open

from data import (
    Fineweb, 
    ChatMLTokenizer, 
    SmolTalk, 
    UltraFeedbackSFT,
    pad_collate, 
    IGNORE_INDEX,
    CHATML_TEMPLATE,
    RESPONSE_HEADER
)


class MockEncoding:
    """Mock for tiktoken.Encoding"""
    def __init__(self, name="mock_tokenizer"):
        self.name = name
        self.eot_token = 1

    def encode_ordinary(self, text):
        """Return a simple encoding for testing"""
        return [i for i in range(10)]

    def encode(self, text, allowed_special=None):
        """Return a simple encoding based on the text"""
        if RESPONSE_HEADER in text:
            return [10, 11, 12]  # Special tokens for header
        return [i for i in range(20)]


class TestFineweb(unittest.TestCase):
    """Test cases for the Fineweb dataset class"""

    def setUp(self):
        """Set up common test fixtures"""
        self.tokenizer = MockEncoding()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock memmap file
        self.mock_memmap_path = os.path.join(self.temp_dir, f"fineweb-sample-10BT-{self.tokenizer.name}.bin")
        mock_data = np.arange(10000, dtype=np.uint16)
        mock_memmap = np.memmap(self.mock_memmap_path, dtype=np.uint16, mode="w+", shape=10000)
        mock_memmap[:] = mock_data
        mock_memmap.flush()

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters"""
        dataset = Fineweb(
            tokenizer=self.tokenizer,
            root_path=self.temp_dir,
            subset="sample-10BT",
            split="train",
            tokens_per_sample=1024,
            samples_per_epoch=10
        )
        
        self.assertEqual(dataset.tokens_per_sample, 1024)
        self.assertEqual(dataset.samples_per_epoch, 10)
        self.assertEqual(dataset.tokenizer, self.tokenizer)

    def test_init_with_invalid_subset(self):
        """Test initialization with invalid subset"""
        with self.assertRaises(ValueError):
            Fineweb(
                tokenizer=self.tokenizer,
                root_path=self.temp_dir,
                subset="invalid-subset"
            )

    def test_init_with_invalid_split(self):
        """Test initialization with invalid split"""
        with self.assertRaises(ValueError):
            Fineweb(
                tokenizer=self.tokenizer,
                root_path=self.temp_dir,
                split="invalid"
            )

    def test_init_with_invalid_tokens_per_sample(self):
        """Test initialization with invalid tokens_per_sample"""
        with self.assertRaises(ValueError):
            Fineweb(
                tokenizer=self.tokenizer,
                root_path=self.temp_dir,
                tokens_per_sample=0
            )

    def test_init_with_invalid_samples_per_epoch(self):
        """Test initialization with invalid samples_per_epoch"""
        with self.assertRaises(ValueError):
            Fineweb(
                tokenizer=self.tokenizer,
                root_path=self.temp_dir,
                samples_per_epoch=0
            )

    def test_tokenize_method(self):
        """Test the tokenize method"""
        dataset = Fineweb(
            tokenizer=self.tokenizer,
            root_path=self.temp_dir
        )
        
        sample = {"text": "This is a test"}
        result = dataset.tokenize(sample)
        
        self.assertEqual(len(result["tokens"]), 11)  # 10 tokens + 1 EOT token
        self.assertEqual(result["tokens"][-1], self.tokenizer.eot_token)

    @patch('random.randint')
    def test_iter_method(self, mock_randint):
        """Test the __iter__ method"""
        mock_randint.return_value = 0
        
        dataset = Fineweb(
            tokenizer=self.tokenizer,
            root_path=self.temp_dir,
            tokens_per_sample=10,
            samples_per_epoch=2
        )
        
        iterator = iter(dataset)
        x, y = next(iterator)
        
        self.assertEqual(len(x), 10)
        self.assertEqual(len(y), 10)
        
        # Check that y is offset by 1
        self.assertEqual(list(y), list(range(1, 11)))
        
        # Check we can get another sample
        x2, y2 = next(iterator)
        self.assertEqual(len(x2), 10)
        
        # Check that the iterator stops after samples_per_epoch
        with self.assertRaises(StopIteration):
            next(iterator)


class TestChatMLTokenizer(unittest.TestCase):
    """Test cases for the ChatMLTokenizer class"""
    
    def setUp(self):
        self.tokenizer = MockEncoding()
        self.chatml_tokenizer = ChatMLTokenizer(
            tokenizer=self.tokenizer,
            max_tokens_per_sample=100
        )

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters"""
        self.assertEqual(self.chatml_tokenizer.tokenizer, self.tokenizer)
        self.assertEqual(self.chatml_tokenizer.max_tokens_per_sample, 100)
        self.assertEqual(self.chatml_tokenizer.response_header_length, 3)  # From our mock

    def test_init_with_invalid_params(self):
        """Test initialization with invalid max_tokens_per_sample"""
        with self.assertRaises(ValueError):
            ChatMLTokenizer(
                tokenizer=self.tokenizer,
                max_tokens_per_sample=0
            )

    def test_tokenize_messages_with_completion(self):
        """Test tokenizing messages with completion (assistant role)"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        x, y = self.chatml_tokenizer.tokenize_messages(messages)
        
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.shape, y.shape)
        
        # Check some tokens have IGNORE_INDEX in y
        self.assertTrue(IGNORE_INDEX in y)
        
        # Check we have fewer than max_tokens
        self.assertLessEqual(len(x), self.chatml_tokenizer.max_tokens_per_sample)

    def test_tokenize_messages_without_completion(self):
        """Test tokenizing messages without completion (no assistant role)"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "I am a system message"}
        ]
        
        # Should trigger a warning
        with self.assertWarns(UserWarning):
            x, y = self.chatml_tokenizer.tokenize_messages(messages)
        
        self.assertEqual(x.shape, y.shape)
        
        # All tokens should be IGNORE_INDEX in labels
        self.assertTrue(all(t == IGNORE_INDEX for t in y))

    def test_tokenize_messages_truncation(self):
        """Test truncation of oversized messages"""
        # Create a message that would generate more tokens than max_tokens_per_sample
        messages = [
            {"role": "user", "content": "A" * 1000},  # Very long message
            {"role": "assistant", "content": "B" * 1000}  # Very long response
        ]
        
        x, y = self.chatml_tokenizer.tokenize_messages(messages)
        
        self.assertLessEqual(len(x), self.chatml_tokenizer.max_tokens_per_sample)
        self.assertLessEqual(len(y), self.chatml_tokenizer.max_tokens_per_sample)


class TestSmolTalk(unittest.TestCase):
    """Test cases for the SmolTalk dataset class"""
    
    def setUp(self):
        self.mock_tokenizer = MockEncoding()
        self.mock_chatml_tokenizer = ChatMLTokenizer(
            tokenizer=self.mock_tokenizer,
            max_tokens_per_sample=100
        )
        
        # Mock the dataset loading
        self.dataset_patcher = patch('data.load_dataset')
        self.mock_dataset_loader = self.dataset_patcher.start()
        
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.__getitem__.return_value = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
        }
        self.mock_dataset_loader.return_value = mock_dataset

    def tearDown(self):
        self.dataset_patcher.stop()

    def test_init_with_valid_params(self):
        """Test initialization with valid subset"""
        dataset = SmolTalk(
            tokenizer=self.mock_chatml_tokenizer,
            subset="all"
        )
        
        self.assertEqual(dataset.tokenizer, self.mock_chatml_tokenizer)
        self.mock_dataset_loader.assert_called_once_with(
            SmolTalk.DATASET_NAME, "all", split="train"
        )

    def test_init_with_invalid_subset(self):
        """Test initialization with invalid subset"""
        with self.assertRaises(ValueError):
            SmolTalk(
                tokenizer=self.mock_chatml_tokenizer,
                subset="invalid-subset"
            )

    def test_getitem_method(self):
        """Test the __getitem__ method"""
        dataset = SmolTalk(tokenizer=self.mock_chatml_tokenizer)
        
        x, y = dataset[0]
        
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.shape, y.shape)

    def test_len_method(self):
        """Test the __len__ method"""
        dataset = SmolTalk(tokenizer=self.mock_chatml_tokenizer)
        
        self.assertEqual(len(dataset), 10)


class TestUltraFeedbackSFT(unittest.TestCase):
    """Test cases for the UltraFeedbackSFT dataset class"""
    
    def setUp(self):
        self.mock_tokenizer = MockEncoding()
        self.mock_chatml_tokenizer = ChatMLTokenizer(
            tokenizer=self.mock_tokenizer,
            max_tokens_per_sample=100
        )
        
        # Mock the dataset loading
        self.dataset_patcher = patch('data.load_dataset')
        self.mock_dataset_loader = self.dataset_patcher.start()
        
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.__getitem__.return_value = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
        }
        self.mock_dataset_loader.return_value = mock_dataset

    def tearDown(self):
        self.dataset_patcher.stop()

    def test_init_with_valid_params(self):
        """Test initialization with valid split"""
        dataset = UltraFeedbackSFT(
            tokenizer=self.mock_chatml_tokenizer,
            split="train"
        )
        
        self.assertEqual(dataset.tokenizer, self.mock_chatml_tokenizer)

    def test_init_with_invalid_split(self):
        """Test initialization with invalid split"""
        with self.assertRaises(ValueError):
            UltraFeedbackSFT(
                tokenizer=self.mock_chatml_tokenizer,
                split="invalid"
            )

    def test_getitem_method(self):
        """Test the __getitem__ method"""
        dataset = UltraFeedbackSFT(tokenizer=self.mock_chatml_tokenizer)
        
        x, y = dataset[0]
        
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.shape, y.shape)

    def test_len_method(self):
        """Test the __len__ method"""
        dataset = UltraFeedbackSFT(tokenizer=self.mock_chatml_tokenizer)
        
        self.assertEqual(len(dataset), 10)


if __name__ == '__main__':
    unittest.main()