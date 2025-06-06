import unittest
import torch
import torch.nn as nn

from torch import Tensor

from math import sqrt

from model import (
    NoPEGPT,
    DecoderBlock,
    SelfAttention,
    MLP,
    LoRA,
    NoPEGPTHuggingFaceConfig,
    NoPEGPTHuggingFaceModel,
)

from caching import KVCache


class TestNoPEGPT(unittest.TestCase):
    """Test cases for the NoPEGPT model."""

    def setUp(self):
        """Set up a small model instance for testing."""
        self.model = NoPEGPT(
            vocabulary_size=1000,
            embedding_dimensions=128,
            num_heads=8,
            num_layers=2,
            feed_forward_ratio=4,
            dropout=0.1,
        )

        # Use CPU for testing
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def test_initialization(self):
        """Test that the model initializes correctly."""
        self.assertEqual(self.model.vocabulary_size, 1000)
        self.assertEqual(self.model.token_embeddings.num_embeddings, 1000)
        self.assertEqual(self.model.token_embeddings.embedding_dim, 128)
        self.assertEqual(len(self.model.body), 2)
        self.assertIsInstance(self.model.output_norm, nn.RMSNorm)
        self.assertIsInstance(self.model.output_layer, nn.Linear)
        self.assertEqual(self.model.output_layer.weight.shape, (1000, 128))

        # Test weight tying
        self.assertTrue(
            torch.all(
                torch.eq(
                    self.model.token_embeddings.weight, self.model.output_layer.weight
                )
            )
        )

    def test_initialization_with_invalid_params(self):
        """Test that initialization fails with invalid parameters."""
        with self.assertRaises(ValueError):
            NoPEGPT(
                vocabulary_size=0,  # Invalid
                embedding_dimensions=128,
                num_heads=8,
                num_layers=2,
                feed_forward_ratio=4,
                dropout=0.1,
            )

        with self.assertRaises(ValueError):
            NoPEGPT(
                vocabulary_size=1000,
                embedding_dimensions=128,
                num_heads=8,
                num_layers=0,  # Invalid
                feed_forward_ratio=4,
                dropout=0.1,
            )

    def test_forward(self):
        """Test the forward pass of the model."""
        batch_size = 2
        seq_len = 10

        # Create input and target tensors
        x = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
        y = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)

        # Perform forward pass
        logits, loss = self.model(x, y)

        # Check output shapes
        self.assertEqual(logits.shape, (batch_size, seq_len, 1000))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.is_tensor(loss))

        # Test forward pass without targets
        logits, loss = self.model(x)
        self.assertEqual(logits.shape, (batch_size, seq_len, 1000))
        self.assertIsNone(loss)

    def test_predict(self):
        """Test the predict method."""
        batch_size = 2
        seq_len = 10

        # Create input tensor
        x = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)

        kv_cache = KVCache(self.model, batch_size, seq_len).to(x.device)

        # Get predictions
        logits = self.model.predict(x, kv_cache)

        # Check output shape (should be [batch_size, vocab_size])
        self.assertEqual(logits.shape, (batch_size, 1000))

    def test_generate(self):
        """Test the generate method."""
        seq_len = 10

        # Create prompt tensor
        prompt = torch.randint(0, 1000, (seq_len,), device=self.device)

        # Set up generation parameters
        max_tokens = 5

        # Generate tokens
        generated_tokens = list(
            self.model.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=1.0,
                top_k=500,
                top_p=0.9,
                repeat_penalty=0.1,
            )
        )

        # Check that the right number of tokens was generated
        self.assertLessEqual(len(generated_tokens), max_tokens)

        # Check that each token is an integer
        for token, probability in generated_tokens:
            self.assertTrue(isinstance(token, Tensor))
            self.assertTrue(isinstance(probability, Tensor))

    def test_generate_with_invalid_params(self):
        """Test that generate fails with invalid parameters."""
        seq_len = 10
        prompt = torch.randint(0, 1000, (seq_len,), device=self.device)

        with self.assertRaises(ValueError):
            next(self.model.generate(prompt, max_tokens=0))

        with self.assertRaises(ValueError):
            next(self.model.generate(prompt, context_length=0))

        with self.assertRaises(ValueError):
            next(self.model.generate(prompt, temperature=0))

        with self.assertRaises(ValueError):
            next(self.model.generate(prompt, top_k=0))

        with self.assertRaises(ValueError):
            next(self.model.generate(prompt, top_p=0))

    def test_num_trainable_params(self):
        """Test the num_trainable_params property."""
        # All parameters should be trainable initially
        initial_params = self.model.num_trainable_params
        self.assertGreater(initial_params, 0)

        # Freeze all parameters
        self.model.freeze_model_parameters()
        self.assertEqual(self.model.num_trainable_params, 0)

        # Unfreeze token embeddings only
        self.model.unfreeze_token_embeddings()
        token_embed_params = sum(
            p.numel() for p in self.model.token_embeddings.parameters()
        )
        self.assertEqual(self.model.num_trainable_params, token_embed_params)

    def test_resize_token_embeddings(self):
        """Test resizing token embeddings."""
        original_vocab_size = self.model.vocabulary_size
        new_vocab_size = original_vocab_size * 2

        # Resize embeddings
        self.model.resize_token_embeddings(new_vocab_size)

        # Check that sizes were updated
        self.assertEqual(self.model.vocabulary_size, new_vocab_size)
        self.assertEqual(self.model.token_embeddings.num_embeddings, new_vocab_size)
        self.assertEqual(self.model.output_layer.weight.shape[0], new_vocab_size)

        # Check that weight tying is preserved
        self.assertTrue(
            torch.all(
                torch.eq(
                    self.model.token_embeddings.weight, self.model.output_layer.weight
                )
            )
        )

        # Test invalid size
        with self.assertRaises(ValueError):
            self.model.resize_token_embeddings(0)

    def test_add_and_merge_lora(self):
        """Test adding LoRA parameters and merging them."""
        # Add LoRA parameters
        rank = 4
        alpha = 1.0
        dropout = 0.1

        self.model.add_lora_parameters(rank, alpha, dropout)

        # Check for LoRA parameters in state dict
        state_dict = self.model.state_dict()
        lora_params = [name for name in state_dict if "lora" in name]
        self.assertGreater(len(lora_params), 0)

        # Get LoRA state dict
        lora_dict = self.model.lora_state_dict()
        self.assertEqual(len(lora_dict), len(lora_params))

        # Save weights before merging
        pre_merge_weights = {}
        for name, param in self.model.named_parameters():
            if not "lora" in name and param.requires_grad:
                pre_merge_weights[name] = param.clone()

        # Merge LoRA parameters
        self.model.merge_lora_parameters()

        # Check that LoRA parameters are no longer in state dict
        merged_lora_params = [
            name for name in self.model.state_dict() if "lora" in name
        ]

        self.assertEqual(len(merged_lora_params), 0)

    def test_activation_checkpointing(self):
        """Test enabling activation checkpointing."""
        # Store the original checkpoint function
        original_checkpoint = self.model.checkpoint

        # Enable activation checkpointing
        self.model.enable_activation_checkpointing()

        # Check that the checkpoint function changed
        self.assertNotEqual(self.model.checkpoint, original_checkpoint)


class TestDecoderBlock(unittest.TestCase):
    """Test cases for the DecoderBlock."""

    def setUp(self):
        """Set up a decoder block for testing."""
        self.block = DecoderBlock(
            embedding_dimensions=128, num_heads=8, feed_forward_ratio=4, dropout=0.1
        )

    def test_initialization(self):
        """Test that the block initializes correctly."""
        self.assertIsInstance(self.block.norm1, nn.RMSNorm)
        self.assertIsInstance(self.block.attention, SelfAttention)
        self.assertIsInstance(self.block.norm2, nn.RMSNorm)
        self.assertIsInstance(self.block.mlp, MLP)

    def test_initialization_with_invalid_params(self):
        """Test that initialization fails with invalid parameters."""
        with self.assertRaises(ValueError):
            DecoderBlock(
                embedding_dimensions=0,  # Invalid
                num_heads=8,
                feed_forward_ratio=4,
                dropout=0.1,
            )

        with self.assertRaises(ValueError):
            DecoderBlock(
                embedding_dimensions=128,
                num_heads=8,
                feed_forward_ratio=3,  # Invalid, must be 1, 2, or 4
                dropout=0.1,
            )

    def test_forward(self):
        """Test the forward pass of the block."""
        batch_size = 2
        seq_len = 10
        embed_dim = 128

        # Create input tensor
        x = torch.randn(batch_size, seq_len, embed_dim)

        # Perform forward pass
        output = self.block(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, embed_dim))


class TestSelfAttention(unittest.TestCase):
    """Test cases for the SelfAttention module."""

    def setUp(self):
        """Set up a self-attention module for testing."""
        self.attention = SelfAttention(
            embedding_dimensions=128, num_heads=8, dropout=0.1
        )

    def test_initialization(self):
        """Test that the module initializes correctly."""
        self.assertIsInstance(self.attention.qkv_proj, nn.Linear)
        self.assertIsInstance(self.attention.out_proj, nn.Linear)
        self.assertEqual(self.attention.qkv_proj.weight.shape, (3 * 128, 128))
        self.assertEqual(self.attention.out_proj.weight.shape, (128, 128))
        self.assertEqual(self.attention.num_heads, 8)
        self.assertEqual(self.attention.head_dimensions, 16)
        self.assertAlmostEqual(self.attention.scale, 1.0 / sqrt(16), places=5)

    def test_initialization_with_invalid_params(self):
        """Test that initialization fails with invalid parameters."""
        with self.assertRaises(ValueError):
            SelfAttention(
                embedding_dimensions=127,  # Not divisible by num_heads
                num_heads=8,
                dropout=0.1,
            )

        with self.assertRaises(ValueError):
            SelfAttention(embedding_dimensions=128, num_heads=0, dropout=0.1)  # Invalid

    def test_forward(self):
        """Test the forward pass of the self-attention module."""
        batch_size = 2
        seq_len = 10
        embed_dim = 128

        # Create input tensor
        x = torch.randn(batch_size, seq_len, embed_dim)

        # Perform forward pass
        output = self.attention(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, embed_dim))


class TestMLP(unittest.TestCase):
    """Test cases for the MLP module."""

    def setUp(self):
        """Set up an MLP module for testing."""
        self.mlp = MLP(embedding_dimensions=128, feed_forward_ratio=4, dropout=0.1)

    def test_initialization(self):
        """Test that the module initializes correctly."""
        self.assertIsInstance(self.mlp.layers, nn.Sequential)
        self.assertEqual(len(self.mlp.layers), 3)
        self.assertIsInstance(self.mlp.layers[0], nn.Linear)
        self.assertIsInstance(self.mlp.layers[1], nn.SiLU)
        self.assertIsInstance(self.mlp.layers[2], nn.Linear)
        self.assertEqual(self.mlp.layers[0].weight.shape, (512, 128))
        self.assertEqual(self.mlp.layers[2].weight.shape, (128, 512))
        self.assertIsInstance(self.mlp.dropout, nn.Dropout1d)

    def test_initialization_with_invalid_params(self):
        """Test that initialization fails with invalid parameters."""

        with self.assertRaises(ValueError):
            MLP(embedding_dimensions=128, feed_forward_ratio=0, dropout=0.1)  # Invalid

    def test_forward(self):
        """Test the forward pass of the MLP module."""
        batch_size = 2
        seq_len = 10
        embed_dim = 128

        # Create input tensor
        x = torch.randn(batch_size, seq_len, embed_dim)

        # Perform forward pass
        output = self.mlp(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, embed_dim))


class TestLoRA(unittest.TestCase):
    """Test cases for the LoRA module."""

    def setUp(self):
        """Set up a LoRA module for testing."""
        self.in_features = 128
        self.out_features = 256
        self.rank = 8
        self.alpha = 1.0
        self.dropout = 0.1

        self.lora = LoRA(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
        )

    def test_initialization(self):
        """Test that the module initializes correctly."""
        self.assertEqual(self.lora.lora_a.shape, (self.rank, self.in_features))
        self.assertEqual(self.lora.lora_b.shape, (self.out_features, self.rank))
        self.assertIsInstance(self.lora.dropout, nn.Dropout1d)
        self.assertEqual(self.lora.alpha, self.alpha)

    def test_initialization_with_invalid_params(self):
        """Test that initialization fails with invalid parameters."""
        with self.assertRaises(ValueError):
            LoRA(
                in_features=self.in_features,
                out_features=self.out_features,
                rank=0,  # Invalid
                alpha=self.alpha,
                dropout=self.dropout,
            )

        with self.assertRaises(ValueError):
            LoRA(
                in_features=self.in_features,
                out_features=self.out_features,
                rank=self.rank,
                alpha=0,  # Invalid
                dropout=self.dropout,
            )

        with self.assertRaises(ValueError):
            LoRA(
                in_features=self.in_features,
                out_features=self.out_features,
                rank=self.rank,
                alpha=self.alpha,
                dropout=-0.1,  # Invalid
            )

    def test_from_linear(self):
        """Test creating a LoRA from a Linear layer."""
        linear = nn.Linear(self.in_features, self.out_features)

        lora = LoRA.from_linear(
            linear=linear, rank=self.rank, alpha=self.alpha, dropout=self.dropout
        )

        self.assertEqual(lora.lora_a.shape, (self.rank, self.in_features))
        self.assertEqual(lora.lora_b.shape, (self.out_features, self.rank))
        self.assertEqual(lora.alpha, self.alpha)

    def test_forward(self):
        """Test the forward pass of the LoRA module."""
        # Perturb the LoRA parameters a bit
        self.lora.lora_a.data += 0.1
        self.lora.lora_b.data -= 0.1

        # Create weight tensor with same shape as expected for parametrization
        weight = torch.randn(self.out_features, self.in_features)

        # Perform forward pass
        output = self.lora(weight)

        # Check output shape
        self.assertEqual(output.shape, weight.shape)

        # Output should be different from input
        self.assertFalse(torch.allclose(output, weight))


class TestNoPEGPTHuggingFaceConfig(unittest.TestCase):
    """Test cases for the NoPEGPTHuggingFaceConfig."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        config = NoPEGPTHuggingFaceConfig()

        self.assertEqual(config.vocabulary_size, 50257)
        self.assertEqual(config.embedding_dimensions, 1024)
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.num_layers, 24)
        self.assertEqual(config.feed_forward_ratio, 4)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.model_type, "nope-gpt")

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        config = NoPEGPTHuggingFaceConfig(
            vocabulary_size=1000,
            embedding_dimensions=128,
            num_heads=8,
            num_layers=2,
            feed_forward_ratio=2,
            dropout=0.2,
        )

        self.assertEqual(config.vocabulary_size, 1000)
        self.assertEqual(config.embedding_dimensions, 128)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.feed_forward_ratio, 2)
        self.assertEqual(config.dropout, 0.2)
        self.assertEqual(config.model_type, "nope-gpt")


class TestNoPEGPTHuggingFaceModel(unittest.TestCase):
    """Test cases for the NoPEGPTHuggingFaceModel."""

    def setUp(self):
        """Set up a model for testing."""
        self.config = NoPEGPTHuggingFaceConfig(
            vocabulary_size=1000,
            embedding_dimensions=128,
            num_heads=8,
            num_layers=2,
            feed_forward_ratio=4,
            dropout=0.1,
            padding_index=0,
        )

        self.model = NoPEGPTHuggingFaceModel(self.config)

        # Use CPU for testing
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def test_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model.model, NoPEGPT)

        # Check that config parameters were passed correctly
        self.assertEqual(self.model.model.vocabulary_size, 1000)
        self.assertEqual(self.model.model.token_embeddings.embedding_dim, 128)
        self.assertEqual(len(self.model.model.body), 2)

    def test_forward(self):
        """Test the forward pass of the model."""
        batch_size = 2
        seq_len = 10

        # Create input and target tensors
        x = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
        y = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)

        # Perform forward pass with labels
        output = self.model(x, y)

        # Check output format
        self.assertIn("logits", output)
        self.assertIn("loss", output)
        self.assertEqual(output["logits"].shape, (batch_size, seq_len, 1000))
        self.assertTrue(torch.is_tensor(output["loss"]))

        # Perform forward pass without labels
        output = self.model(x)

        # Check output format
        self.assertIn("logits", output)
        self.assertIn("loss", output)
        self.assertEqual(output["logits"].shape, (batch_size, seq_len, 1000))
        self.assertIsNone(output["loss"])


if __name__ == "__main__":
    unittest.main()
