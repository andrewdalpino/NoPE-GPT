import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from torch import Tensor

from src.nope_gpt.model import (
    NoPEGPT,
    Decoder,
    DecoderBlock,
    SelfAttention,
    InvertedBottleneck,
    TokenClassifier,
    LoRA,
)
from src.nope_gpt.caching import KVCache, DynamicKVBlock
from src.nope_gpt.search import Candidate


class TestLoRA(unittest.TestCase):
    def test_init(self):
        in_features = 10
        out_features = 20
        rank = 4
        alpha = 0.5

        lora = LoRA(in_features, out_features, rank, alpha)

        self.assertEqual(lora.alpha, alpha)
        self.assertEqual(lora.lora_a.shape, (rank, in_features))
        self.assertEqual(lora.lora_b.shape, (out_features, rank))
        # Check lora_b is initialized with zeros
        self.assertTrue(torch.all(lora.lora_b == 0))

    def test_from_linear(self):
        in_features = 10
        out_features = 20
        rank = 4
        alpha = 0.5

        linear = nn.Linear(in_features, out_features, bias=False)

        lora = LoRA.from_linear(linear, rank, alpha)

        self.assertEqual(lora.alpha, alpha)
        self.assertEqual(lora.lora_a.shape, (rank, in_features))
        self.assertEqual(lora.lora_b.shape, (out_features, rank))

    def test_forward(self):
        in_features = 10
        out_features = 20
        rank = 4
        alpha = 0.5

        lora = LoRA(in_features, out_features, rank, alpha)

        # Set specific values for testing
        lora.lora_a.data = torch.ones((rank, in_features))
        lora.lora_b.data = torch.ones((out_features, rank))

        weight = torch.zeros((out_features, in_features))

        result = lora.forward(weight)

        # Check dimensions
        self.assertEqual(result.shape, (out_features, in_features))

        # Check the formula: result = weight + alpha * (lora_b @ lora_a)
        # Since lora_a and lora_b are all ones, and rank=4, each element of lora_b @ lora_a should be 4
        # Then multiplied by alpha (0.5), each element should be 2
        # Since weight is all zeros, the result should be all 2s
        expected = torch.ones((out_features, in_features)) * rank * alpha
        self.assertTrue(torch.allclose(result, expected))


class TestTokenClassifier(unittest.TestCase):
    def test_init(self):
        embedding_dimensions = 32
        vocabulary_size = 100

        classifier = TokenClassifier(embedding_dimensions, vocabulary_size)

        self.assertIsInstance(classifier.norm, nn.RMSNorm)
        self.assertEqual(classifier.norm.normalized_shape, (embedding_dimensions,))

        self.assertIsInstance(classifier.linear, nn.Linear)
        self.assertEqual(classifier.linear.in_features, embedding_dimensions)
        self.assertEqual(classifier.linear.out_features, vocabulary_size)

    def test_forward(self):
        batch_size = 2
        seq_len = 5
        embedding_dimensions = 32
        vocabulary_size = 100

        classifier = TokenClassifier(embedding_dimensions, vocabulary_size)

        # Create a dummy input tensor
        x = torch.randn(batch_size, seq_len, embedding_dimensions)

        # Forward pass
        output = classifier.forward(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, vocabulary_size))


class TestInvertedBottleneck(unittest.TestCase):
    def test_init(self):
        embedding_dimensions = 32
        hidden_ratio = 2
        dropout = 0.1

        bottleneck = InvertedBottleneck(embedding_dimensions, hidden_ratio, dropout)

        self.assertIsInstance(bottleneck.linear1, nn.Linear)
        self.assertEqual(bottleneck.linear1.in_features, embedding_dimensions)
        self.assertEqual(
            bottleneck.linear1.out_features, embedding_dimensions * hidden_ratio
        )

        self.assertIsInstance(bottleneck.linear2, nn.Linear)
        self.assertEqual(
            bottleneck.linear2.in_features, embedding_dimensions * hidden_ratio
        )
        self.assertEqual(bottleneck.linear2.out_features, embedding_dimensions)

        self.assertIsInstance(bottleneck.silu, nn.SiLU)
        self.assertIsInstance(bottleneck.dropout, nn.Dropout1d)
        self.assertEqual(bottleneck.dropout.p, dropout)

    def test_init_with_invalid_hidden_ratio(self):
        with self.assertRaises(AssertionError):
            InvertedBottleneck(32, 3, 0.1)  # hidden_ratio must be 1, 2, or 4

    def test_forward(self):
        batch_size = 2
        seq_len = 5
        embedding_dimensions = 32
        hidden_ratio = 2
        dropout = 0.0  # Use 0 to make testing deterministic

        bottleneck = InvertedBottleneck(embedding_dimensions, hidden_ratio, dropout)

        # Create a dummy input tensor
        x = torch.randn(batch_size, seq_len, embedding_dimensions)

        # Forward pass
        output = bottleneck.forward(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, embedding_dimensions))

    def test_predict(self):
        batch_size = 2
        seq_len = 5
        embedding_dimensions = 32
        hidden_ratio = 2
        dropout = 0.1

        bottleneck = InvertedBottleneck(embedding_dimensions, hidden_ratio, dropout)

        # Create a dummy input tensor
        x = torch.randn(batch_size, seq_len, embedding_dimensions)

        # Predict pass (should not apply dropout)
        output = bottleneck.predict(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, embedding_dimensions))


class TestSelfAttention(unittest.TestCase):
    def test_init(self):
        embedding_dimensions = 64
        num_q_heads = 4
        num_kv_heads = 2
        dropout = 0.1

        attention = SelfAttention(
            embedding_dimensions, num_q_heads, num_kv_heads, dropout
        )

        self.assertEqual(attention.embedding_dimensions, embedding_dimensions)
        self.assertEqual(attention.num_q_heads, num_q_heads)
        self.assertEqual(attention.num_kv_heads, num_kv_heads)
        self.assertEqual(attention.head_dimensions, embedding_dimensions // num_q_heads)
        self.assertEqual(attention.dropout, dropout)
        self.assertEqual(
            attention.scale, 1.0 / (embedding_dimensions // num_q_heads) ** 0.5
        )
        self.assertTrue(attention.is_gqa)  # Since num_q_heads > num_kv_heads

        self.assertIsInstance(attention.q_proj, nn.Linear)
        self.assertIsInstance(attention.k_proj, nn.Linear)
        self.assertIsInstance(attention.v_proj, nn.Linear)
        self.assertIsInstance(attention.out_proj, nn.Linear)

        # Check projection dimensions
        self.assertEqual(attention.q_proj.in_features, embedding_dimensions)
        self.assertEqual(attention.q_proj.out_features, embedding_dimensions)

        head_dimensions = embedding_dimensions // num_q_heads
        kv_dimensions = num_kv_heads * head_dimensions

        self.assertEqual(attention.k_proj.out_features, kv_dimensions)
        self.assertEqual(attention.v_proj.out_features, kv_dimensions)

    def test_init_invalid_params(self):
        # Test num_q_heads < num_kv_heads (not allowed)
        with self.assertRaises(AssertionError):
            SelfAttention(64, 2, 4, 0.1)

        # Test embedding_dimensions not divisible by num_q_heads
        with self.assertRaises(AssertionError):
            SelfAttention(65, 4, 2, 0.1)

    @patch("src.nope_gpt.model.scaled_dot_product_attention")
    def test_forward(self, mock_attention):
        batch_size = 2
        seq_len = 5
        embedding_dimensions = 64
        num_q_heads = 4
        num_kv_heads = 2
        dropout = 0.1

        attention = SelfAttention(
            embedding_dimensions, num_q_heads, num_kv_heads, dropout
        )

        # Mock the attention mechanism
        mock_attention.return_value = torch.randn(
            batch_size, num_q_heads, seq_len, embedding_dimensions // num_q_heads
        )

        # Create a dummy input tensor
        x = torch.randn(batch_size, seq_len, embedding_dimensions)

        # Forward pass
        output = attention.forward(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, embedding_dimensions))

        # Verify attention was called
        mock_attention.assert_called_once()

    @patch("src.nope_gpt.model.scaled_dot_product_attention")
    def test_predict(self, mock_attention):
        batch_size = 2
        seq_len = 1  # Autoregressive phase has seq_len=1
        embedding_dimensions = 64
        num_q_heads = 4
        num_kv_heads = 2
        dropout = 0.1

        attention = SelfAttention(
            embedding_dimensions, num_q_heads, num_kv_heads, dropout
        )

        # Create a mock KV block
        mock_kv_block = MagicMock(spec=DynamicKVBlock)
        mock_kv_block.update.return_value = (
            torch.randn(
                batch_size, num_kv_heads, 10, embedding_dimensions // num_q_heads
            ),  # k
            torch.randn(
                batch_size, num_kv_heads, 10, embedding_dimensions // num_q_heads
            ),  # v
        )

        # Mock the attention mechanism
        mock_attention.return_value = torch.randn(
            batch_size, num_q_heads, seq_len, embedding_dimensions // num_q_heads
        )

        # Create a dummy input tensor
        x = torch.randn(batch_size, seq_len, embedding_dimensions)

        # Predict pass
        output = attention.predict(x, mock_kv_block)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, embedding_dimensions))

        # Verify kv_block.update was called
        mock_kv_block.update.assert_called_once()

        # Verify attention was called with is_causal=False for autoregressive phase
        args, kwargs = mock_attention.call_args
        self.assertFalse(kwargs.get("is_causal", True))


class TestDecoderBlock(unittest.TestCase):
    def test_init(self):
        embedding_dimensions = 64
        num_q_heads = 4
        num_kv_heads = 2
        hidden_ratio = 2
        dropout = 0.1

        block = DecoderBlock(
            embedding_dimensions, num_q_heads, num_kv_heads, hidden_ratio, dropout
        )

        self.assertIsInstance(block.stage1, SelfAttention)
        self.assertIsInstance(block.stage2, InvertedBottleneck)
        self.assertIsInstance(block.norm1, nn.RMSNorm)
        self.assertIsInstance(block.norm2, nn.RMSNorm)

    def test_forward(self):
        batch_size = 2
        seq_len = 5
        embedding_dimensions = 64
        num_q_heads = 4
        num_kv_heads = 2
        hidden_ratio = 2
        dropout = 0.0  # Use 0 to make testing deterministic

        block = DecoderBlock(
            embedding_dimensions, num_q_heads, num_kv_heads, hidden_ratio, dropout
        )

        # Create a dummy input tensor
        x = torch.randn(batch_size, seq_len, embedding_dimensions)

        # Forward pass
        output = block.forward(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, embedding_dimensions))

    def test_predict(self):
        batch_size = 2
        seq_len = 1
        embedding_dimensions = 64
        num_q_heads = 4
        num_kv_heads = 2
        hidden_ratio = 2
        dropout = 0.1

        block = DecoderBlock(
            embedding_dimensions, num_q_heads, num_kv_heads, hidden_ratio, dropout
        )

        # Create a mock KV block
        mock_kv_block = MagicMock(spec=DynamicKVBlock)

        # Patch the predict methods of stage1 and stage2
        with (
            patch.object(
                block.stage1,
                "predict",
                return_value=torch.randn(batch_size, seq_len, embedding_dimensions),
            ),
            patch.object(
                block.stage2,
                "predict",
                return_value=torch.randn(batch_size, seq_len, embedding_dimensions),
            ),
        ):

            # Create a dummy input tensor
            x = torch.randn(batch_size, seq_len, embedding_dimensions)

            # Predict pass
            output = block.predict(x, mock_kv_block)

            # Check output shape
            self.assertEqual(output.shape, (batch_size, seq_len, embedding_dimensions))

            # Verify stage1.predict was called with kv_block
            block.stage1.predict.assert_called_once()
            args, kwargs = block.stage1.predict.call_args
            self.assertEqual(len(args), 2)
            self.assertIs(args[1], mock_kv_block)


class TestDecoder(unittest.TestCase):
    def test_init(self):
        embedding_dimensions = 64
        num_q_heads = 4
        num_kv_heads = 2
        num_layers = 3
        hidden_ratio = 2
        dropout = 0.1

        decoder = Decoder(
            embedding_dimensions,
            num_q_heads,
            num_kv_heads,
            num_layers,
            hidden_ratio,
            dropout,
        )

        self.assertIsInstance(decoder.layers, nn.ModuleList)
        self.assertEqual(len(decoder.layers), num_layers)
        for layer in decoder.layers:
            self.assertIsInstance(layer, DecoderBlock)

    def test_init_invalid_num_layers(self):
        with self.assertRaises(AssertionError):
            Decoder(64, 4, 2, 0, 2, 0.1)  # num_layers must be > 0

    def test_forward(self):
        batch_size = 2
        seq_len = 5
        embedding_dimensions = 64
        num_q_heads = 4
        num_kv_heads = 2
        num_layers = 2
        hidden_ratio = 2
        dropout = 0.0  # Use 0 to make testing deterministic

        decoder = Decoder(
            embedding_dimensions,
            num_q_heads,
            num_kv_heads,
            num_layers,
            hidden_ratio,
            dropout,
        )

        # Create a dummy input tensor
        x = torch.randn(batch_size, seq_len, embedding_dimensions)

        # Forward pass
        output = decoder.forward(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, embedding_dimensions))

    def test_predict(self):
        batch_size = 2
        seq_len = 1
        embedding_dimensions = 64
        num_q_heads = 4
        num_kv_heads = 2
        num_layers = 2
        hidden_ratio = 2
        dropout = 0.1
        context_length = 10

        decoder = Decoder(
            embedding_dimensions,
            num_q_heads,
            num_kv_heads,
            num_layers,
            hidden_ratio,
            dropout,
        )

        # Create a mock KV cache
        mock_kv_cache = MagicMock(spec=KVCache)
        mock_kv_blocks = [MagicMock(spec=DynamicKVBlock) for _ in range(num_layers)]

        # Make the KV cache iterable to yield the blocks
        mock_kv_cache.__iter__.return_value = iter(mock_kv_blocks)

        # Patch predict method of each layer
        with (
            patch.object(
                decoder.layers[0],
                "predict",
                return_value=torch.randn(batch_size, seq_len, embedding_dimensions),
            ),
            patch.object(
                decoder.layers[1],
                "predict",
                return_value=torch.randn(batch_size, seq_len, embedding_dimensions),
            ),
        ):

            # Create a dummy input tensor
            x = torch.randn(batch_size, seq_len, embedding_dimensions)

            # Predict pass
            output = decoder.predict(x, mock_kv_cache)

            # Check output shape
            self.assertEqual(output.shape, (batch_size, seq_len, embedding_dimensions))

            # Verify each layer.predict was called
            for i, layer in enumerate(decoder.layers):
                layer.predict.assert_called_once()
                args, kwargs = layer.predict.call_args
                self.assertEqual(len(args), 2)
                # Check the kv_block was passed correctly to each layer
                self.assertIs(args[1], mock_kv_blocks[i])


class TestNoPEGPT(unittest.TestCase):
    def test_init(self):
        vocabulary_size = 100
        embedding_dimensions = 64
        num_q_heads = 4
        num_kv_heads = 2
        num_decoder_layers = 2
        hidden_ratio = 2
        dropout = 0.1

        model = NoPEGPT(
            vocabulary_size,
            embedding_dimensions,
            num_q_heads,
            num_kv_heads,
            num_decoder_layers,
            hidden_ratio,
            dropout,
        )

        self.assertIsInstance(model.token_embeddings, nn.Embedding)
        self.assertEqual(model.token_embeddings.num_embeddings, vocabulary_size)
        self.assertEqual(model.token_embeddings.embedding_dim, embedding_dimensions)

        self.assertIsInstance(model.decoder, Decoder)
        self.assertEqual(len(model.decoder.layers), num_decoder_layers)

        self.assertIsInstance(model.token_classifier, TokenClassifier)

        # Check that weights are tied
        self.assertIs(
            model.token_classifier.linear.weight, model.token_embeddings.weight
        )

    def test_init_invalid_params(self):
        # Test vocabulary_size <= 0
        with self.assertRaises(AssertionError):
            NoPEGPT(0, 64, 4, 2, 2, 2, 0.1)

        # Test embedding_dimensions <= 0
        with self.assertRaises(AssertionError):
            NoPEGPT(100, 0, 4, 2, 2, 2, 0.1)

        # Test num_decoder_layers <= 0
        with self.assertRaises(AssertionError):
            NoPEGPT(100, 64, 4, 2, 0, 2, 0.1)

    def test_num_trainable_params(self):
        vocabulary_size = 100
        embedding_dimensions = 64
        num_q_heads = 4
        num_kv_heads = 2
        num_decoder_layers = 2
        hidden_ratio = 2
        dropout = 0.1

        model = NoPEGPT(
            vocabulary_size,
            embedding_dimensions,
            num_q_heads,
            num_kv_heads,
            num_decoder_layers,
            hidden_ratio,
            dropout,
        )

        # Calculate expected number of parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertEqual(model.num_trainable_params, num_params)

    def test_freeze_model_parameters(self):
        vocabulary_size = 100
        embedding_dimensions = 64
        num_q_heads = 4
        num_kv_heads = 2
        num_decoder_layers = 2
        hidden_ratio = 2
        dropout = 0.1

        model = NoPEGPT(
            vocabulary_size,
            embedding_dimensions,
            num_q_heads,
            num_kv_heads,
            num_decoder_layers,
            hidden_ratio,
            dropout,
        )

        # Initially all parameters should be trainable
        for param in model.parameters():
            self.assertTrue(param.requires_grad)

        # Freeze parameters
        model.freeze_model_parameters()

        # Now all parameters should be frozen
        for param in model.parameters():
            self.assertFalse(param.requires_grad)

    def test_unfreeze_token_embeddings(self):
        vocabulary_size = 100
        embedding_dimensions = 64
        num_q_heads = 4
        num_kv_heads = 2
        num_decoder_layers = 2
        hidden_ratio = 2
        dropout = 0.1

        model = NoPEGPT(
            vocabulary_size,
            embedding_dimensions,
            num_q_heads,
            num_kv_heads,
            num_decoder_layers,
            hidden_ratio,
            dropout,
        )

        # Freeze all parameters
        model.freeze_model_parameters()

        # All parameters should be frozen
        for param in model.parameters():
            self.assertFalse(param.requires_grad)

        # Unfreeze token embeddings
        model.unfreeze_token_embeddings()

        # Only token embeddings should be trainable
        self.assertTrue(model.token_embeddings.weight.requires_grad)

        # Other parameters should still be frozen
        for name, param in model.named_parameters():
            if name != "token_embeddings.weight":
                self.assertFalse(param.requires_grad)

    def test_forward(self):
        batch_size = 2
        seq_len = 5
        vocabulary_size = 100
        embedding_dimensions = 64
        num_q_heads = 4
        num_kv_heads = 2
        num_decoder_layers = 2
        hidden_ratio = 2
        dropout = 0.0  # Use 0 to make testing deterministic

        model = NoPEGPT(
            vocabulary_size,
            embedding_dimensions,
            num_q_heads,
            num_kv_heads,
            num_decoder_layers,
            hidden_ratio,
            dropout,
        )

        # Create a dummy input tensor (token indices)
        x = torch.randint(0, vocabulary_size, (batch_size, seq_len))

        # Forward pass
        output = model.forward(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, vocabulary_size))

    @patch("src.nope_gpt.model.KVCache")
    def test_predict(self, mock_kv_cache_class):
        batch_size = 1
        seq_len = 1
        vocabulary_size = 100
        embedding_dimensions = 64
        num_q_heads = 4
        num_kv_heads = 2
        num_decoder_layers = 2
        hidden_ratio = 2
        dropout = 0.1

        model = NoPEGPT(
            vocabulary_size,
            embedding_dimensions,
            num_q_heads,
            num_kv_heads,
            num_decoder_layers,
            hidden_ratio,
            dropout,
        )

        # Create a mock KV cache
        mock_kv_cache = MagicMock(spec=KVCache)

        # Patch decoder.predict to return a known tensor
        with patch.object(
            model.decoder,
            "predict",
            return_value=torch.randn(batch_size, seq_len, embedding_dimensions),
        ):

            # Create a dummy input tensor (token indices)
            x = torch.randint(0, vocabulary_size, (batch_size, seq_len))

            # Predict pass
            output = model.predict(x, mock_kv_cache)

            # Check output shape - should be (batch_size, vocabulary_size) since we pluck the last token
            self.assertEqual(output.shape, (batch_size, vocabulary_size))

            # Verify decoder.predict was called with the kv_cache
            model.decoder.predict.assert_called_once()
            args, kwargs = model.decoder.predict.call_args
            self.assertEqual(len(args), 2)
            self.assertIs(args[1], mock_kv_cache)


if __name__ == "__main__":
    unittest.main()
