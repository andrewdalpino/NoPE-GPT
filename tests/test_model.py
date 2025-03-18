import unittest

import torch
import torch.nn as nn

from model import (
    LightGPT,
    CausalSelfAttentionBlock,
    MLP,
    LoRA,
    LightGPTHuggingFaceConfig,
    LightGPTHuggingFaceModel,
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
        )

    def test_init_with_invalid_vocabulary_size(self):
        with self.assertRaises(ValueError):
            LightGPT(
                vocabulary_size=0,
                embedding_dimensions=128,
                num_heads=8,
                num_layers=2,
                feed_forward_ratio=4,
                dropout=0.1,
                padding_index=0,
            )

    def test_init_with_invalid_num_layers(self):
        with self.assertRaises(ValueError):
            LightGPT(
                vocabulary_size=1000,
                embedding_dimensions=128,
                num_heads=8,
                num_layers=0,
                feed_forward_ratio=4,
                dropout=0.1,
                padding_index=0,
            )

    def test_init_with_invalid_feed_forward_ratio(self):
        with self.assertRaises(ValueError):
            LightGPT(
                vocabulary_size=1000,
                embedding_dimensions=128,
                num_heads=8,
                num_layers=2,
                feed_forward_ratio=3,  # Invalid, must be 1, 2, or 4
                dropout=0.1,
                padding_index=0,
            )

    def test_forward(self):
        x = torch.randint(0, 1000, (2, 10))
        y = torch.randint(0, 1000, (2, 10))

        output, loss = self.model(x, y)

        self.assertEqual(output.shape, (2, 10, 1000))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.is_tensor(loss))

    def test_forward_no_labels(self):
        x = torch.randint(0, 1000, (2, 10))

        output, loss = self.model(x)

        self.assertEqual(output.shape, (2, 10, 1000))
        self.assertIsNone(loss)

    def test_predict(self):
        x = torch.randint(0, 1000, (2, 10))

        output = self.model.predict(x)

        self.assertEqual(output.shape, (2, 1000))

    def test_generate(self):
        prompt = torch.randint(0, 1000, (10,))

        tokens = list(self.model.generate(prompt, max_tokens=5))

        self.assertLessEqual(len(tokens), 5)
        for token in tokens:
            self.assertTrue(isinstance(token, torch.Tensor))

    def test_generate_with_invalid_params(self):
        prompt = torch.randint(0, 1000, (10,))

        with self.assertRaises(ValueError):
            list(self.model.generate(prompt, max_tokens=0))

        with self.assertRaises(ValueError):
            list(self.model.generate(prompt, temperature=0))

        with self.assertRaises(ValueError):
            list(self.model.generate(prompt, top_k=0))

        with self.assertRaises(ValueError):
            list(self.model.generate(prompt, top_p=0))

    def test_beam_search(self):
        prompt = torch.randint(0, 1000, (10,))

        candidates = self.model.beam_search(prompt, max_tokens=5, num_candidates=2)

        self.assertEqual(len(candidates), 2)
        for candidate in candidates:
            self.assertTrue(hasattr(candidate, "tokens"))
            self.assertTrue(hasattr(candidate, "cumulative_log_probability"))

    def test_beam_search_with_invalid_params(self):
        prompt = torch.randint(0, 1000, (10,))

        with self.assertRaises(ValueError):
            self.model.beam_search(prompt, max_tokens=0)

        with self.assertRaises(ValueError):
            self.model.beam_search(prompt, num_candidates=0)

        with self.assertRaises(ValueError):
            self.model.beam_search(prompt, beam_width=0)

        with self.assertRaises(ValueError):
            self.model.beam_search(prompt, length_penalty=0)

    def test_num_trainable_params(self):
        num_params = self.model.num_trainable_params
        self.assertGreater(num_params, 0)

        # Freeze all parameters
        self.model.freeze_model_parameters()
        self.assertEqual(self.model.num_trainable_params, 0)

    def test_freeze_and_unfreeze(self):
        # Freeze model parameters
        self.model.freeze_model_parameters()
        for param in self.model.parameters():
            self.assertFalse(param.requires_grad)

        # Unfreeze token embeddings
        self.model.unfreeze_token_embeddings()
        for param in self.model.token_embeddings.parameters():
            self.assertTrue(param.requires_grad)

    def test_resize_token_embeddings(self):
        original_vocab_size = self.model.vocabulary_size
        new_vocab_size = original_vocab_size * 2

        self.model.resize_token_embeddings(new_vocab_size)

        self.assertEqual(self.model.vocabulary_size, new_vocab_size)
        self.assertEqual(self.model.token_embeddings.num_embeddings, new_vocab_size)
        self.assertEqual(self.model.output_layer.weight.shape[0], new_vocab_size)

        with self.assertRaises(ValueError):
            self.model.resize_token_embeddings(0)

    def test_add_lora_parameters(self):
        rank = 4
        alpha = 16
        dropout = 0.1

        # Add LoRA parameters
        self.model.add_lora_parameters(rank, alpha, dropout)

        # Check that LoRA parameters exist in state_dict
        state_dict = self.model.state_dict()
        lora_params = [name for name in state_dict if "lora" in name]
        self.assertGreater(len(lora_params), 0)

        # Test lora_state_dict
        lora_dict = self.model.lora_state_dict()
        self.assertEqual(len(lora_dict), len(lora_params))

        # Test merge_lora_parameters
        self.model.merge_lora_parameters()


class TestCausalSelfAttentionBlock(unittest.TestCase):
    def setUp(self):
        self.block = CausalSelfAttentionBlock(
            embedding_dimensions=128,
            num_heads=8,
            feed_forward_ratio=4,
            dropout=0.1,
        )

    def test_init_with_invalid_params(self):
        with self.assertRaises(ValueError):
            CausalSelfAttentionBlock(
                embedding_dimensions=0,
                num_heads=8,
                feed_forward_ratio=4,
                dropout=0.1,
            )

        with self.assertRaises(ValueError):
            CausalSelfAttentionBlock(
                embedding_dimensions=128,
                num_heads=0,
                feed_forward_ratio=4,
                dropout=0.1,
            )

        with self.assertRaises(ValueError):
            CausalSelfAttentionBlock(
                embedding_dimensions=128,
                num_heads=8,
                feed_forward_ratio=4,
                dropout=1.1,  # Invalid, must be between 0 and 1
            )

    def test_forward(self):
        batch_size = 2
        seq_len = 10
        embed_dim = 128

        x = torch.randn(batch_size, seq_len, embed_dim)
        attention_mask = torch.full((seq_len, seq_len), float("-inf"))
        attention_mask = torch.triu(attention_mask, diagonal=1)

        output = self.block(x, attention_mask)

        self.assertEqual(output.shape, (batch_size, seq_len, embed_dim))


class TestMLP(unittest.TestCase):
    def setUp(self):
        self.mlp = MLP(
            embedding_dimensions=128,
            hidden_dimensions=512,
            dropout=0.1,
        )

    def test_init_with_invalid_params(self):
        with self.assertRaises(ValueError):
            MLP(
                embedding_dimensions=0,
                hidden_dimensions=512,
                dropout=0.1,
            )

        with self.assertRaises(ValueError):
            MLP(
                embedding_dimensions=128,
                hidden_dimensions=0,
                dropout=0.1,
            )

    def test_forward(self):
        batch_size = 2
        seq_len = 10
        embed_dim = 128

        x = torch.randn(batch_size, seq_len, embed_dim)
        output = self.mlp(x)

        self.assertEqual(output.shape, (batch_size, seq_len, embed_dim))


class TestLoRA(unittest.TestCase):
    def setUp(self):
        self.in_features = 128
        self.out_features = 256
        self.rank = 8
        self.alpha = 16
        self.dropout = 0.1

        self.lora = LoRA(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
        )

    def test_init_with_invalid_params(self):
        with self.assertRaises(ValueError):
            LoRA(
                in_features=self.in_features,
                out_features=self.out_features,
                rank=0,  # Invalid, must be greater than 0
                alpha=self.alpha,
                dropout=self.dropout,
            )

        with self.assertRaises(ValueError):
            LoRA(
                in_features=self.in_features,
                out_features=self.out_features,
                rank=self.rank,
                alpha=0,  # Invalid, must be greater than 0
                dropout=self.dropout,
            )

        with self.assertRaises(ValueError):
            LoRA(
                in_features=self.in_features,
                out_features=self.out_features,
                rank=self.rank,
                alpha=self.alpha,
                dropout=-0.1,  # Invalid, must be between 0 and 1
            )

    def test_from_linear(self):
        linear = nn.Linear(self.in_features, self.out_features)

        lora = LoRA.from_linear(
            linear=linear,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
        )

        self.assertEqual(lora.lora_a.shape, (self.rank, self.in_features))
        self.assertEqual(lora.lora_b.shape, (self.out_features, self.rank))

    def test_forward(self):
        weight = torch.randn(self.out_features, self.in_features)
        output = self.lora(weight)

        self.assertEqual(output.shape, weight.shape)


class TestLightGPTHuggingFaceConfig(unittest.TestCase):
    def test_init_default(self):
        config = LightGPTHuggingFaceConfig()

        self.assertEqual(config.vocabulary_size, 50257)
        self.assertEqual(config.embedding_dimensions, 1024)
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.num_layers, 24)
        self.assertEqual(config.feed_forward_ratio, 4)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.padding_index, -100)
        self.assertEqual(config.model_type, "lightgpt")

    def test_init_custom(self):
        config = LightGPTHuggingFaceConfig(
            vocabulary_size=1000,
            embedding_dimensions=512,
            num_heads=8,
            num_layers=12,
            feed_forward_ratio=2,
            dropout=0.2,
            padding_index=0,
        )

        self.assertEqual(config.vocabulary_size, 1000)
        self.assertEqual(config.embedding_dimensions, 512)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.num_layers, 12)
        self.assertEqual(config.feed_forward_ratio, 2)
        self.assertEqual(config.dropout, 0.2)
        self.assertEqual(config.padding_index, 0)


class TestLightGPTHuggingFaceModel(unittest.TestCase):
    def setUp(self):
        self.config = LightGPTHuggingFaceConfig(
            vocabulary_size=1000,
            embedding_dimensions=128,
            num_heads=8,
            num_layers=2,
            feed_forward_ratio=4,
            dropout=0.1,
            padding_index=0,
        )

        self.model = LightGPTHuggingFaceModel(self.config)

    def test_init(self):
        self.assertIsInstance(self.model.model, LightGPT)

    def test_forward(self):
        x = torch.randint(0, 1000, (2, 10))
        y = torch.randint(0, 1000, (2, 10))

        output = self.model(x, y)

        self.assertIn("logits", output)
        self.assertIn("loss", output)
        self.assertEqual(output["logits"].shape, (2, 10, 1000))
        self.assertTrue(torch.is_tensor(output["loss"]))

        output = self.model(x)

        self.assertIn("logits", output)
        self.assertIn("loss", output)
        self.assertEqual(output["logits"].shape, (2, 10, 1000))
        self.assertIsNone(output["loss"])


if __name__ == "__main__":
    unittest.main()
