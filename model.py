from math import sqrt

import torch

from torch import Tensor
from torch.nn import (
    Module,
    ModuleList,
    Sequential,
    Embedding,
    MultiheadAttention,
    Linear,
    LayerNorm,
    GELU,
    Dropout1d,
    CrossEntropyLoss,
    Parameter,
    Buffer,
)

from torch.nn.functional import softmax
from torch.nn.init import normal_

from typing import Iterator


class GPT(Module):
    """A generative pre-trained transformer."""

    EOS_INDEX = 50256

    PADDING_INDEX = 50257

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimensions: int,
        block_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        if vocabulary_size <= 0:
            raise ValueError(
                f"Vocabulary size must be greater than 0, {vocabulary_size} given."
            )

        if num_layers <= 0:
            raise ValueError(f"Num layers must be greater than 0, {num_layers} given.")

        token_embeddings = Embedding(
            vocabulary_size, embedding_dimensions, padding_idx=self.PADDING_INDEX
        )

        positional_embeddings = Embedding(block_size, embedding_dimensions)

        output_layer = Linear(embedding_dimensions, vocabulary_size, bias=False)

        token_embeddings.weight = output_layer.weight  # Tie weights

        self.token_embeddings = token_embeddings
        self.positional_embeddings = positional_embeddings
        self.body = ModuleList(
            [
                CausalSelfAttentionBlock(
                    embedding_dimensions, block_size, num_heads, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = LayerNorm(embedding_dimensions, bias=False)
        self.output_layer = output_layer

        positions = torch.arange(0, block_size)

        causal_mask = torch.tril(torch.ones((block_size, block_size)))
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float("-Inf"))
        causal_mask = causal_mask.masked_fill(causal_mask == 1, 0.0)

        self.positions = Buffer(positions, persistent=False)
        self.causal_mask = Buffer(causal_mask, persistent=False)

        self.loss_function = CrossEntropyLoss(ignore_index=self.PADDING_INDEX)

        self.vocabulary_size = vocabulary_size
        self.block_size = block_size

    @property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def forward(
        self, x: Tensor, y: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        b, t = x.size()

        positions = self.positions[:t]

        causal_mask = self.causal_mask[:t, :t]

        tok_out = self.token_embeddings(x)
        pos_out = self.positional_embeddings(positions)

        z = tok_out + pos_out

        for layer in self.body:
            z = layer(z, causal_mask)

        z = self.output_norm(z)
        z = self.output_layer(z)

        if y is not None:
            y_pred = z.view(-1, z.size(-1))
            labels = y.view(-1)

            loss = self.loss_function(y_pred, labels)

        else:
            loss = None

        return z, loss

    @torch.no_grad()
    def generate(
        self, prompt: Tensor, max_tokens: int, temperature: float, top_k: int
    ) -> Iterator:
        """Given a prompt, sample the next {max_tokens} tokens from the model."""

        if max_tokens <= 0:
            raise ValueError(f"Max tokens must be greater than 0, {max_tokens} given.")

        if temperature <= 0:
            raise ValueError(
                f"Temperature must be greater than 0, {temperature} given."
            )

        if top_k > self.vocabulary_size:
            raise ValueError(
                f"Top k must be less than the vocabulary size, {top_k} given."
            )

        context_window = prompt

        for i in range(max_tokens):
            context_window = context_window[-self.block_size :]

            y_pred, _ = self.forward(context_window.unsqueeze(0))

            y_pred = y_pred[:, -1, :].squeeze(0)

            logits, indices = torch.topk(y_pred, top_k, sorted=False)

            logits /= temperature

            probabilities = softmax(logits, dim=-1)

            offset = torch.multinomial(probabilities, num_samples=1).squeeze(0)

            next_token = indices[offset]

            if next_token == self.EOS_INDEX:
                break

            yield next_token

            if i < max_tokens:
                new_context = next_token.unsqueeze(0)

                context_window = torch.cat((context_window, new_context))


class CausalSelfAttentionBlock(Module):
    """Causal self-attention block with residual connections."""

    def __init__(
        self, embedding_dimensions: int, block_size: int, num_heads: int, dropout: float
    ):
        super().__init__()

        if embedding_dimensions <= 0:
            raise ValueError(
                f"Embedding dimensions must be greater than 0, {embedding_dimensions} given."
            )

        if block_size <= 0:
            raise ValueError(f"Block size must be greater than 0, {block_size} given.")

        if num_heads <= 0:
            raise ValueError(f"Num heads must be greater than 0, {num_heads} given.")

        if dropout < 0 or dropout > 1:
            raise ValueError(f"Dropout must be between 0 and 1, {dropout} given")

        self.norm1 = LayerNorm(embedding_dimensions, bias=False)
        self.attention = MultiheadAttention(
            embedding_dimensions,
            num_heads,
            batch_first=True,
            bias=False,
            dropout=dropout,
        )

        self.norm2 = LayerNorm(embedding_dimensions, bias=False)
        self.mlp = MLP(embedding_dimensions, 4 * embedding_dimensions, dropout)

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        b, t, c = x.size()

        z = self.norm1(x)
        z, _ = self.attention(z, z, z, attn_mask=causal_mask, is_causal=True)

        z = x + z  # Residual connection

        x = z

        z = self.norm2(x)
        z = self.mlp(z)

        z = x + z  # Residual connection

        return z


class MLP(Module):
    """A two-layer fully-connected network with dropout."""

    def __init__(
        self, embedding_dimensions: int, hidden_dimensions: int, dropout: float
    ):
        super().__init__()

        if embedding_dimensions <= 0:
            raise ValueError(
                f"Embedding dimensions must be greater than 0, {embedding_dimensions} given."
            )

        if hidden_dimensions <= 0:
            raise ValueError(
                f"Hidden dimensions must be greater than 0, {hidden_dimensions} given."
            )

        self.layers = Sequential(
            Linear(embedding_dimensions, hidden_dimensions, bias=False),
            GELU(),
            Linear(hidden_dimensions, embedding_dimensions, bias=False),
            Dropout1d(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers.forward(x)


class GPTWithLoRA(Module):
    """A LoRA wrapper for pre-trained models."""

    def __init__(self, model: GPT, rank: int, alpha: float):
        super().__init__()

        for param in model.parameters():
            param.requires_grad = False

        for module in model.body:
            for i, layer in enumerate(module.mlp.layers):
                if isinstance(layer, Linear):
                    module.mlp.layers[i] = LoRALinear(layer, rank, alpha)

        model.output_layer = LoRALinear(model.output_layer, rank, alpha)

        self.model = model

    @property
    def num_trainable_params(self) -> int:
        return self.model.num_trainable_params

    def state_dict(self):
        return {
            name: module
            for name, module in super().state_dict().items()
            if "lora" in name
        }

    def forward(
        self, x: Tensor, y: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        return self.model.forward(x, y)

    def generate(
        self, prompts: Tensor, max_tokens: int, temperature: float, top_k: int
    ) -> Tensor:
        return self.model.generate(prompts, max_tokens, temperature, top_k)


class LoRALinear(Module):
    """Adapter layer for injecting LoRA into linear layers."""

    def __init__(self, linear: Linear, rank: int, alpha: float):
        super().__init__()

        self.linear = linear

        self.lora = LoRA(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x) + self.lora(x)


class LoRA(Module):
    """Rank decomposition layer."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()

        if in_features <= 0:
            raise ValueError(
                f"In features must be greater than 0, {in_features} given."
            )

        if out_features <= 0:
            raise ValueError(
                f"Out features must be greater than 0, {out_features} given."
            )

        if rank <= 0:
            raise ValueError(f"Rank must be greater than 0, {rank} given.")

        if alpha < 0.0:
            raise ValueError(f"Alpha must be greater than 0, {alpha} given.")

        std_dev = 1.0 / sqrt(rank)

        self.a = Parameter(torch.randn(in_features, rank) * std_dev)
        self.b = Parameter(torch.zeros(rank, out_features))

        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        z = x @ self.a @ self.b

        z *= self.alpha

        return z
