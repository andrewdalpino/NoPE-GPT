from typing import Self, Iterator

import torch

from torch import Tensor
from torch.nn import Module, ModuleList, Buffer


class KVCache(Module):
    """Key-value cache for all layers of the model."""

    def __init__(self, model: Module, batch_size: int, context_length: int):
        super().__init__()

        from model import SelfAttention

        self.kv_blocks = ModuleList(
            [
                DynamicKVBlock(
                    batch_size,
                    layer.embedding_dimensions,
                    layer.num_heads,
                    context_length,
                )
                for layer in model.modules()
                if isinstance(layer, SelfAttention)
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

        self.k = Buffer(
            torch.empty(batch_size, num_heads, 0, head_dimensions),
            persistent=False,
        )

        self.v = Buffer(
            torch.empty(batch_size, num_heads, 0, head_dimensions),
            persistent=False,
        )

        self.context_length = context_length

    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        self.k = torch.cat((self.k, k), dim=2)
        self.v = torch.cat((self.v, v), dim=2)

        if self.k.size(2) > self.context_length:
            self.k = self.k[:, :, -self.context_length :, :]
            self.v = self.v[:, :, -self.context_length :, :]

        return self.k, self.v
