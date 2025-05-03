import unittest
import torch

from torch.nn import Buffer

from caching import KVCache, DynamicKVBlock


class MockAttention:
    """Mock for attention module in a transformer layer"""

    def __init__(self, embedding_dimensions, num_heads):
        self.embedding_dimensions = embedding_dimensions
        self.num_heads = num_heads


class MockLayer:
    """Mock for a transformer layer with attention"""

    def __init__(self, embedding_dimensions, num_heads):
        self.attention = MockAttention(embedding_dimensions, num_heads)


class MockModel:
    """Mock for a model with transformer layers"""

    def __init__(self):
        self.body = [MockLayer(128, 8), MockLayer(128, 8)]


class TestDynamicKVBlock(unittest.TestCase):
    """Test cases for the DynamicKVBlock class."""

    def setUp(self):
        """Set up a DynamicKVBlock instance for testing."""
        self.batch_size = 2
        self.embedding_dimensions = 128
        self.num_heads = 8
        self.context_length = 100
        self.head_dimensions = self.embedding_dimensions // self.num_heads

        self.kv_block = DynamicKVBlock(
            batch_size=self.batch_size,
            embedding_dimensions=self.embedding_dimensions,
            num_heads=self.num_heads,
            context_length=self.context_length,
        )

    def test_initialization(self):
        """Test that the block initializes correctly."""
        self.assertIsInstance(self.kv_block.k_cache, Buffer)
        self.assertIsInstance(self.kv_block.v_cache, Buffer)

        # Check initial cache shapes
        self.assertEqual(
            self.kv_block.k_cache.shape,
            (self.batch_size, self.num_heads, 0, self.head_dimensions),
        )
        self.assertEqual(
            self.kv_block.v_cache.shape,
            (self.batch_size, self.num_heads, 0, self.head_dimensions),
        )

        # Check context_length is set
        self.assertEqual(self.kv_block.context_length, self.context_length)

    def test_initialization_with_invalid_params(self):
        """Test initialization with invalid parameters."""
        # Test invalid batch size
        with self.assertRaises(ValueError):
            DynamicKVBlock(
                0, self.embedding_dimensions, self.num_heads, self.context_length
            )

        # Test invalid embedding dimensions
        with self.assertRaises(ValueError):
            DynamicKVBlock(self.batch_size, 0, self.num_heads, self.context_length)

        # Test invalid number of heads
        with self.assertRaises(ValueError):
            DynamicKVBlock(
                self.batch_size, self.embedding_dimensions, 0, self.context_length
            )

        # Test embedding dimensions not divisible by num_heads
        with self.assertRaises(ValueError):
            DynamicKVBlock(self.batch_size, 127, self.num_heads, self.context_length)

        # Test invalid context length
        with self.assertRaises(ValueError):
            DynamicKVBlock(
                self.batch_size, self.embedding_dimensions, self.num_heads, 0
            )

    def test_update(self):
        """Test updating the cache with new key-value pairs."""
        # Create test tensors
        seq_len = 10
        k = torch.randn(self.batch_size, self.num_heads, seq_len, self.head_dimensions)
        v = torch.randn(self.batch_size, self.num_heads, seq_len, self.head_dimensions)

        # Update cache
        k_cache, v_cache = self.kv_block.update(k, v)

        # Check shapes after update
        self.assertEqual(
            k_cache.shape,
            (self.batch_size, self.num_heads, seq_len, self.head_dimensions),
        )
        self.assertEqual(
            v_cache.shape,
            (self.batch_size, self.num_heads, seq_len, self.head_dimensions),
        )

        # Check that the updated cache contains our values
        self.assertTrue(torch.allclose(k_cache, k))
        self.assertTrue(torch.allclose(v_cache, v))

        # Check that the internal buffers were updated
        self.assertTrue(torch.allclose(self.kv_block.k_cache, k))
        self.assertTrue(torch.allclose(self.kv_block.v_cache, v))

    def test_update_multiple_times(self):
        """Test updating the cache multiple times."""
        # First update
        seq_len1 = 10
        k1 = torch.randn(
            self.batch_size, self.num_heads, seq_len1, self.head_dimensions
        )
        v1 = torch.randn(
            self.batch_size, self.num_heads, seq_len1, self.head_dimensions
        )

        self.kv_block.update(k1, v1)

        # Second update
        seq_len2 = 15
        k2 = torch.randn(
            self.batch_size, self.num_heads, seq_len2, self.head_dimensions
        )
        v2 = torch.randn(
            self.batch_size, self.num_heads, seq_len2, self.head_dimensions
        )

        k_cache, v_cache = self.kv_block.update(k2, v2)

        # Check shapes after update (should be seq_len1 + seq_len2)
        self.assertEqual(
            k_cache.shape,
            (
                self.batch_size,
                self.num_heads,
                seq_len1 + seq_len2,
                self.head_dimensions,
            ),
        )
        self.assertEqual(
            v_cache.shape,
            (
                self.batch_size,
                self.num_heads,
                seq_len1 + seq_len2,
                self.head_dimensions,
            ),
        )

        # Check that the updated cache contains both updates in the right order
        self.assertTrue(torch.allclose(k_cache[:, :, :seq_len1], k1))
        self.assertTrue(torch.allclose(k_cache[:, :, seq_len1:], k2))
        self.assertTrue(torch.allclose(v_cache[:, :, :seq_len1], v1))
        self.assertTrue(torch.allclose(v_cache[:, :, seq_len1:], v2))

    def test_truncation(self):
        """Test truncation when cache exceeds context length."""
        # Create a KV block with small context length
        small_context = 20
        kv_block = DynamicKVBlock(
            batch_size=self.batch_size,
            embedding_dimensions=self.embedding_dimensions,
            num_heads=self.num_heads,
            context_length=small_context,
        )

        # First update with exactly context_length tokens
        k1 = torch.randn(
            self.batch_size, self.num_heads, small_context, self.head_dimensions
        )
        v1 = torch.randn(
            self.batch_size, self.num_heads, small_context, self.head_dimensions
        )

        kv_block.update(k1, v1)

        # Second update with more tokens
        seq_len2 = 10
        k2 = torch.randn(
            self.batch_size, self.num_heads, seq_len2, self.head_dimensions
        )
        v2 = torch.randn(
            self.batch_size, self.num_heads, seq_len2, self.head_dimensions
        )

        k_cache, v_cache = kv_block.update(k2, v2)

        # Check shapes after update (should be context_length)
        self.assertEqual(
            k_cache.shape,
            (self.batch_size, self.num_heads, small_context, self.head_dimensions),
        )
        self.assertEqual(
            v_cache.shape,
            (self.batch_size, self.num_heads, small_context, self.head_dimensions),
        )

        # Check that older tokens were dropped and newer tokens were kept
        # The last small_context-seq_len2 tokens from k1 and all of k2 should be present
        self.assertTrue(
            torch.allclose(
                k_cache[:, :, : (small_context - seq_len2)], k1[:, :, seq_len2:]
            )
        )
        self.assertTrue(torch.allclose(k_cache[:, :, (small_context - seq_len2) :], k2))


class TestKVCache(unittest.TestCase):
    """Test cases for the KVCache class."""

    def setUp(self):
        """Set up a KVCache instance for testing."""
        self.model = MockModel()
        self.batch_size = 2
        self.context_length = 100

        self.kv_cache = KVCache(
            model=self.model,
            batch_size=self.batch_size,
            context_length=self.context_length,
        )

    def test_initialization(self):
        """Test that the cache initializes correctly."""
        # Check that KVCache created the right number of blocks
        self.assertEqual(len(self.kv_cache.kv_blocks), len(self.model.body))

        # Check that each block is a DynamicKVBlock
        for block in self.kv_cache.kv_blocks:
            self.assertIsInstance(block, DynamicKVBlock)

            # Check that the block was initialized with the right parameters
            self.assertEqual(block.k_cache.shape[0], self.batch_size)
            self.assertEqual(block.context_length, self.context_length)

    def test_iteration(self):
        """Test iterating over the cache blocks."""
        # Test that __iter__ works
        blocks = list(self.kv_cache)

        # Check that we got the right number of blocks
        self.assertEqual(len(blocks), len(self.model.body))

        # Check that each block is a DynamicKVBlock
        for block in blocks:
            self.assertIsInstance(block, DynamicKVBlock)


if __name__ == "__main__":
    unittest.main()
