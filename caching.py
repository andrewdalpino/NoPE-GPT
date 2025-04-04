from typing import Self, Iterator

import torch

from torch import Tensor
from torch.nn import Module, ModuleList, Buffer

from model import LightGPT


class KVCache(Module):
    """Key-value cache for all layers of the model."""

    def __init__(self, model: LightGPT, batch_size: int, context_length: int):
        super().__init__()

        self.kv_blocks = ModuleList(
            [
                DynamicKVBlock(
                    batch_size,
                    layer.attention.embedding_dimensions,
                    layer.attention.num_heads,
                    context_length,
                )
                for layer in model.body
            ]
        )

    def __iter__(self) -> Iterator["DynamicKVBlock"]:
        for kv_block in self.kv_blocks:
            yield kv_block


class DynamicKVBlock(Module):
    """A key-value block for a single layer with dynamic memory allocation."""

    def __init__(
        self,
        batch_size: int,
        embedding_dimensions: int,
        num_heads: int,
        context_length: int,
    ):
        super().__init__()

        if batch_size <= 0:
            raise ValueError(f"Batch size must be greater than 0, {batch_size} given.")

        if embedding_dimensions <= 0:
            raise ValueError(
                f"Embedding dimensions must be greater than 0, {embedding_dimensions} given."
            )

        if num_heads <= 0:
            raise ValueError(f"Num heads must be greater than 0, {num_heads} given.")

        if embedding_dimensions % num_heads != 0:
            raise ValueError(
                f"Embedding dimensions must be divisible by num heads, {embedding_dimensions} and {num_heads} given."
            )

        if context_length <= 0:
            raise ValueError(
                f"Context length must be greater than 0, {context_length} given."
            )

        head_dimensions = embedding_dimensions // num_heads

        k_cache = torch.empty(batch_size, num_heads, 0, head_dimensions)
        v_cache = torch.empty(batch_size, num_heads, 0, head_dimensions)

        self.k_cache = Buffer(k_cache, persistent=False)
        self.v_cache = Buffer(v_cache, persistent=False)

        self.context_length = context_length

    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Update the cache with a new key-value pairs.
        Args:
            k (Tensor): Key tensor of shape (batch_size, num_heads, seq_len, head_dimensions).
            v (Tensor): Value tensor of shape (batch_size, num_heads, seq_len, head_dimensions).
        Returns:
            tuple[Tensor, Tensor]: Updated key and value caches.
        """

        k_cache = torch.cat((self.k_cache, k), dim=2)
        v_cache = torch.cat((self.v_cache, v), dim=2)

        if self.k_cache.size(2) > self.context_length:
            k_cache = self.k_cache[:, :, -self.context_length :]
            v_cache = self.v_cache[:, :, -self.context_length :]

        self.k_cache.data = k_cache
        self.v_cache.data = v_cache

        return self.k_cache, self.v_cache
