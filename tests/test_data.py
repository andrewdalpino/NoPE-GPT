import unittest

from unittest.mock import patch, MagicMock

from data import Fineweb, SmolTalk

from tiktoken import Encoding

class TestFineweb(unittest.TestCase):
    @patch('data.load_dataset')
    @patch('os.path.exists', return_value=False)
    @patch('numpy.memmap')
    def test_init(self, mock_memmap, mock_exists, mock_load_dataset):
        tokenizer = MagicMock(spec=Encoding)
        tokenizer.name = "r50k_base"
        tokenizer.encode_ordinary.return_value = [1, 2, 3]
        tokenizer.eot_token = 4

        mock_load_dataset.return_value = MagicMock()
        mock_load_dataset.return_value.map.return_value.train_test_split.return_value = {
            "train": MagicMock(),
            "test": MagicMock()
        }

        dataset = Fineweb(
            tokenizer=tokenizer,
            root_path="./dataset",
            subset="sample-10BT",
            split="train",
            tokens_per_sample=1024,
            samples_per_epoch=4096,
            num_processes=8,
        )

        self.assertEqual(dataset.tokenizer, tokenizer)
        self.assertEqual(dataset.tokens_per_sample, 1024)
        self.assertEqual(dataset.samples_per_epoch, 4096)
        self.assertTrue(mock_exists.called)
        self.assertTrue(mock_load_dataset.called)
        self.assertTrue(mock_memmap.called)

    def test_invalid_subset(self):
        tokenizer = MagicMock(spec=Encoding)

        with self.assertRaises(ValueError):
            Fineweb(tokenizer=tokenizer, subset="invalid_subset")

    def test_invalid_split(self):
        tokenizer = MagicMock(spec=Encoding)

        with self.assertRaises(ValueError):
            Fineweb(tokenizer=tokenizer, split="invalid_split")

    def test_invalid_tokens_per_sample(self):
        tokenizer = MagicMock(spec=Encoding)

        with self.assertRaises(ValueError):
            Fineweb(tokenizer=tokenizer, tokens_per_sample=0)

    def test_invalid_samples_per_epoch(self):
        tokenizer = MagicMock(spec=Encoding)

        with self.assertRaises(ValueError):
            Fineweb(tokenizer=tokenizer, samples_per_epoch=0)


class TestSmolTalk(unittest.TestCase):
    @patch('data.load_dataset')
    def test_init(self, mock_load_dataset):
        tokenizer = MagicMock(spec=Encoding)
        tokenizer.name = "r50k_base"
        tokenizer.n_vocab = 150257
        tokenizer._pat_str = ""

        tokenizer._mergeable_ranks = {
            b"IQ": 0,
            b"Ig": 1,
            b"Iw": 2,
            b"JA": 3,
            b"JQ": 4,
        }
        
        tokenizer._special_tokens = {"<|endoftext|>": 50256}

        mock_load_dataset.return_value = MagicMock()

        dataset = SmolTalk(
            tokenizer=tokenizer,
            subset="all",
            max_tokens_per_sample=1024,
        )

        self.assertEqual(dataset.tokenizer.name, "r50k_base")
        self.assertEqual(dataset.max_tokens_per_sample, 1024)
        self.assertTrue(mock_load_dataset.called)

    def test_invalid_subset(self):
        tokenizer = MagicMock(spec=Encoding)

        with self.assertRaises(ValueError):
            SmolTalk(tokenizer=tokenizer, subset="invalid_subset")

    def test_invalid_max_tokens_per_sample(self):
        tokenizer = MagicMock(spec=Encoding)

        with self.assertRaises(ValueError):
            SmolTalk(tokenizer=tokenizer, max_tokens_per_sample=0)


if __name__ == "__main__":
    unittest.main()