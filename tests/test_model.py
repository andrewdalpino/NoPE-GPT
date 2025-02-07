import unittest

import torch

from model import (
    LightGPT,
    LightGPTInstruct,
    ONNXModel,
    CausalSelfAttentionBlock,
    MLP,
    LoRA,
)


class TestLightGPT(unittest.TestCase):
    def setUp(self):
        self.model = LightGPT(
            vocabulary_size=1000,
            embedding_dimensions=128,
            num_heads=8,
            num_layers=2,
            feed_forward_ratio=4,
            dropout=0.1,
            padding_index=0,
            eos_index=1,
        )

    def test_forward(self):
        x = torch.randint(0, 1000, (2, 10))
        y = torch.randint(0, 1000, (2, 10))

        output, loss = self.model(x, y)

        self.assertEqual(output.shape, (2, 10, 1000))
        self.assertIsNotNone(loss)

    def test_predict(self):
        x = torch.randint(0, 1000, (2, 10))

        output = self.model.predict(x)

        self.assertEqual(output.shape, (2, 1000))

    def test_generate(self):
        prompt = torch.randint(0, 1000, (10,))

        tokens = list(self.model.generate(prompt, max_tokens=5))

        self.assertLessEqual(len(tokens), 5)

    def test_beam_search(self):
        prompt = torch.randint(0, 1000, (10,))

        candidates = self.model.beam_search(prompt, max_tokens=5, num_candidates=2)

        self.assertEqual(len(candidates), 2)


class TestLightGPTInstruct(unittest.TestCase):
    def setUp(self):
        base_model = LightGPT(
            vocabulary_size=1000,
            embedding_dimensions=128,
            num_heads=8,
            num_layers=2,
            feed_forward_ratio=4,
            dropout=0.1,
            padding_index=0,
            eos_index=1,
        )

        self.model = LightGPTInstruct(base_model, rank=4, alpha=1.0, dropout=0.1)

    def test_forward(self):
        x = torch.randint(0, 1000, (2, 10))
        y = torch.randint(0, 1000, (2, 10))

        output, loss = self.model(x, y)

        self.assertEqual(output.shape, (2, 10, 1000))
        self.assertIsNotNone(loss)

    def test_predict(self):
        x = torch.randint(0, 1000, (2, 10))

        output = self.model.predict(x)

        self.assertEqual(output.shape, (2, 1000))


class TestONNXModel(unittest.TestCase):
    def setUp(self):
        base_model = LightGPT(
            vocabulary_size=1000,
            embedding_dimensions=128,
            num_heads=8,
            num_layers=2,
            feed_forward_ratio=4,
            dropout=0.1,
            padding_index=0,
            eos_index=1,
        )

        self.model = ONNXModel(base_model)

    def test_forward(self):
        x = torch.randint(0, 1000, (2, 10))

        output = self.model(x)

        self.assertEqual(output.shape, (2, 1000))


class TestCausalSelfAttentionBlock(unittest.TestCase):
    def setUp(self):
        self.block = CausalSelfAttentionBlock(
            embedding_dimensions=128,
            num_heads=8,
            feed_forward_ratio=4,
            dropout=0.1,
        )

    def test_forward(self):
        x = torch.randn(2, 10, 128)

        attention_mask = torch.zeros(10, 10)

        output = self.block(x, attention_mask)

        self.assertEqual(output.shape, (2, 10, 128))


class TestMLP(unittest.TestCase):
    def setUp(self):
        self.mlp = MLP(embedding_dimensions=128, hidden_dimensions=512, dropout=0.1)

    def test_forward(self):
        x = torch.randn(2, 10, 128)

        output = self.mlp(x)

        self.assertEqual(output.shape, (2, 10, 128))


class TestLoRA(unittest.TestCase):
    def setUp(self):
        self.lora = LoRA(
            in_features=128, out_features=128, rank=4, alpha=1.0, dropout=0.1
        )

    def test_forward(self):
        x = torch.randn(128, 128)

        output = self.lora(x)

        self.assertEqual(output.shape, (128, 128))


if __name__ == "__main__":
    unittest.main()
