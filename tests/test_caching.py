import unittest

import torch

from torch.nn import Module, ModuleList

from src.nope_gpt.caching import KVCache, DynamicKVBlock


class MockDecoderLayer(Module):
    def __init__(self, embedding_dimensions, num_q_heads, num_kv_heads):
        super().__init__()
        self.stage1 = MockStage(embedding_dimensions, num_q_heads, num_kv_heads)


class MockStage(Module):
    def __init__(self, embedding_dimensions, num_q_heads, num_kv_heads):
        super().__init__()
        self.embedding_dimensions = embedding_dimensions
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads


class MockDecoder(Module):
    def __init__(self, num_layers, embedding_dimensions, num_q_heads, num_kv_heads):
        super().__init__()
        self.layers = ModuleList(
            [
                MockDecoderLayer(embedding_dimensions, num_q_heads, num_kv_heads)
                for _ in range(num_layers)
            ]
        )


class TestDynamicKVBlock(unittest.TestCase):
    def setUp(self):
        """Set up test parameters."""
        self.batch_size = 2
        self.embedding_dimensions = 32
        self.num_q_heads = 4
        self.num_kv_heads = 2
        self.context_length = 8
        self.head_dimensions = self.embedding_dimensions // self.num_q_heads

        self.kv_block = DynamicKVBlock(
            self.batch_size,
            self.embedding_dimensions,
            self.num_q_heads,
            self.num_kv_heads,
            self.context_length,
        )

    def test_init(self):
        self.assertEqual(self.kv_block.k_cache.size(0), self.batch_size)
        self.assertEqual(self.kv_block.k_cache.size(1), self.num_kv_heads)
        self.assertEqual(self.kv_block.k_cache.size(2), 0)  # Empty cache initially
        self.assertEqual(self.kv_block.k_cache.size(3), self.head_dimensions)

        self.assertEqual(self.kv_block.v_cache.size(0), self.batch_size)
        self.assertEqual(self.kv_block.v_cache.size(1), self.num_kv_heads)
        self.assertEqual(self.kv_block.v_cache.size(2), 0)  # Empty cache initially
        self.assertEqual(self.kv_block.v_cache.size(3), self.head_dimensions)

        self.assertEqual(self.kv_block.context_length, self.context_length)

    def test_init_invalid_parameters(self):
        with self.assertRaises(AssertionError):
            DynamicKVBlock(
                0,
                self.embedding_dimensions,
                self.num_q_heads,
                self.num_kv_heads,
                self.context_length,
            )

        with self.assertRaises(AssertionError):
            DynamicKVBlock(
                self.batch_size,
                0,
                self.num_q_heads,
                self.num_kv_heads,
                self.context_length,
            )

        with self.assertRaises(AssertionError):
            DynamicKVBlock(
                self.batch_size,
                self.embedding_dimensions,
                self.num_q_heads,
                0,
                self.context_length,
            )

        with self.assertRaises(AssertionError):
            DynamicKVBlock(
                self.batch_size,
                self.embedding_dimensions,
                self.num_q_heads,
                self.num_kv_heads,
                0,
            )

        # Test that embedding_dimensions must be divisible by num_kv_heads
        with self.assertRaises(AssertionError):
            DynamicKVBlock(
                self.batch_size,
                33,
                self.num_q_heads,
                self.num_kv_heads,
                self.context_length,
            )

    def test_update(self):
        seq_len = 3

        # Create test key and value tensors
        k = torch.randn(
            self.batch_size, self.num_kv_heads, seq_len, self.head_dimensions
        )
        v = torch.randn(
            self.batch_size, self.num_kv_heads, seq_len, self.head_dimensions
        )

        # Update cache
        k_cache, v_cache = self.kv_block.update(k, v)

        # Check cache sizes
        self.assertEqual(k_cache.size(0), self.batch_size)
        self.assertEqual(k_cache.size(1), self.num_kv_heads)
        self.assertEqual(k_cache.size(2), seq_len)  # Cache now has seq_len tokens
        self.assertEqual(k_cache.size(3), self.head_dimensions)

        self.assertEqual(v_cache.size(0), self.batch_size)
        self.assertEqual(v_cache.size(1), self.num_kv_heads)
        self.assertEqual(v_cache.size(2), seq_len)  # Cache now has seq_len tokens
        self.assertEqual(v_cache.size(3), self.head_dimensions)

        # Check that k and v are in the cache
        self.assertTrue(torch.allclose(k_cache, k))
        self.assertTrue(torch.allclose(v_cache, v))

        # Update again with new tokens
        k2 = torch.randn(
            self.batch_size, self.num_kv_heads, seq_len, self.head_dimensions
        )
        v2 = torch.randn(
            self.batch_size, self.num_kv_heads, seq_len, self.head_dimensions
        )

        k_cache2, v_cache2 = self.kv_block.update(k2, v2)

        # Check cache sizes after second update
        self.assertEqual(
            k_cache2.size(2), seq_len * 2
        )  # Cache now has 2*seq_len tokens
        self.assertEqual(
            v_cache2.size(2), seq_len * 2
        )  # Cache now has 2*seq_len tokens

        # Check that both sets of tokens are in the cache
        self.assertTrue(torch.allclose(k_cache2[:, :, :seq_len, :], k))
        self.assertTrue(torch.allclose(k_cache2[:, :, seq_len:, :], k2))
        self.assertTrue(torch.allclose(v_cache2[:, :, :seq_len, :], v))
        self.assertTrue(torch.allclose(v_cache2[:, :, seq_len:, :], v2))

    def test_context_length_limit(self):
        # Create test keys and values that will exceed context_length when combined
        seq_len = self.context_length // 2 + 1

        # First update
        k1 = torch.randn(
            self.batch_size, self.num_kv_heads, seq_len, self.head_dimensions
        )
        v1 = torch.randn(
            self.batch_size, self.num_kv_heads, seq_len, self.head_dimensions
        )
        self.kv_block.update(k1, v1)

        # Second update - this should cause the cache to exceed context_length
        k2 = torch.randn(
            self.batch_size, self.num_kv_heads, seq_len, self.head_dimensions
        )
        v2 = torch.randn(
            self.batch_size, self.num_kv_heads, seq_len, self.head_dimensions
        )
        k_cache, v_cache = self.kv_block.update(k2, v2)

        # Check that cache is truncated to context_length
        self.assertEqual(k_cache.size(2), self.context_length)
        self.assertEqual(v_cache.size(2), self.context_length)

        # Verify the cache contains the most recent tokens
        # Calculate how many tokens from k1 should still be in the cache
        remaining_k1_tokens = max(0, self.context_length - seq_len)
        k1_start_idx = seq_len - remaining_k1_tokens

        if remaining_k1_tokens > 0:
            self.assertTrue(
                torch.allclose(
                    k_cache[:, :, :remaining_k1_tokens, :], k1[:, :, k1_start_idx:, :]
                )
            )
            self.assertTrue(
                torch.allclose(
                    v_cache[:, :, :remaining_k1_tokens, :], v1[:, :, k1_start_idx:, :]
                )
            )

        # Verify k2 tokens are in the cache
        self.assertTrue(
            torch.allclose(
                k_cache[:, :, remaining_k1_tokens:, :],
                k2[:, :, : self.context_length - remaining_k1_tokens, :],
            )
        )
        self.assertTrue(
            torch.allclose(
                v_cache[:, :, remaining_k1_tokens:, :],
                v2[:, :, : self.context_length - remaining_k1_tokens, :],
            )
        )


class TestKVCache(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_layers = 3
        self.embedding_dimensions = 32
        self.num_q_heads = 4
        self.num_kv_heads = 2
        self.context_length = 8

        self.decoder = MockDecoder(
            self.num_layers,
            self.embedding_dimensions,
            self.num_q_heads,
            self.num_kv_heads,
        )

        self.kv_cache = KVCache(self.decoder, self.batch_size, self.context_length)

    def test_init(self):
        # Check that kv_blocks has the correct number of blocks
        self.assertEqual(len(self.kv_cache.kv_blocks), self.num_layers)

        # Check that each block is a DynamicKVBlock with the correct parameters
        for block in self.kv_cache.kv_blocks:
            self.assertIsInstance(block, DynamicKVBlock)
            self.assertEqual(block.k_cache.size(0), self.batch_size)
            self.assertEqual(block.k_cache.size(1), self.num_kv_heads)
            self.assertEqual(block.context_length, self.context_length)

    def test_iteration(self):
        blocks = list(self.kv_cache)

        # Check that iteration yields all blocks
        self.assertEqual(len(blocks), self.num_layers)

        # Check that all blocks are DynamicKVBlocks
        for block in blocks:
            self.assertIsInstance(block, DynamicKVBlock)


if __name__ == "__main__":
    unittest.main()
