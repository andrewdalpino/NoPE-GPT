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
from torch.nn.utils.parametrize import register_parametrization, remove_parametrizations

from typing import Iterator, Self


class GPT(Module):
    """A generative pre-trained transformer."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimensions: int,
        block_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        padding_index: int,
        eos_index: int,
    ):
        super().__init__()

        if vocabulary_size <= 0:
            raise ValueError(
                f"Vocabulary size must be greater than 0, {vocabulary_size} given."
            )

        if num_layers <= 0:
            raise ValueError(f"Num layers must be greater than 0, {num_layers} given.")

        token_embeddings = Embedding(
            vocabulary_size, embedding_dimensions, padding_idx=padding_index
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

        self.loss_function = CrossEntropyLoss(ignore_index=padding_index)

        self.vocabulary_size = vocabulary_size
        self.block_size = block_size
        self.eos_index = eos_index

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

            logits = y_pred[0, -1, :]

            logits, indices = torch.topk(logits, top_k, sorted=False)

            if temperature != 1.0:
                logits /= temperature

            probabilities = softmax(logits, dim=0)

            offset = torch.multinomial(probabilities, num_samples=1).squeeze(0)

            next_token = indices[offset]

            if next_token == self.eos_index:
                break

            yield next_token

            new_context = next_token.unsqueeze(0)

            context_window = torch.cat([context_window, new_context])


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
            dropout=dropout,
            bias=False,
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
            out_features, in_features = module.attention.in_proj_weight.shape

            register_parametrization(
                module.attention,
                "in_proj_weight",
                LoRA(in_features, out_features, rank, alpha),
            )

            register_parametrization(
                module.attention.out_proj,
                "weight",
                LoRA.from_linear(module.attention.out_proj, rank, alpha),
            )

            for layer in module.mlp.layers:
                if isinstance(layer, Linear):
                    register_parametrization(
                        layer,
                        "weight",
                        LoRA.from_linear(layer, rank, alpha),
                    ),

        register_parametrization(
            model.output_layer,
            "weight",
            LoRA.from_linear(model.output_layer, rank, alpha),
        ),

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

    def merge_parameters(self) -> GPT:
        """Merge the LoRA parameters with the original parameters."""

        for module in self.model.modules():
            if hasattr(module, "parametrizations"):
                for name in module.parametrizations.keys():
                    remove_parametrizations(module, name, leave_parametrized=True)

        return self.model

    def forward(
        self, x: Tensor, y: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        return self.model.forward(x, y)

    def generate(
        self, prompts: Tensor, max_tokens: int, temperature: float, top_k: int
    ) -> Iterator:
        return self.model.generate(prompts, max_tokens, temperature, top_k)


class LoRA(Module):
    """Rank decomposition transformation."""

    @classmethod
    def from_linear(cls, linear: Linear, rank: int, alpha: float) -> Self:
        out_features, in_features = linear.weight.shape

        return cls(in_features, out_features, rank, alpha)

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

        if alpha <= 0.0:
            raise ValueError(f"Alpha must be greater than 0, {alpha} given.")

        std_dev = 1.0 / sqrt(rank)

        self.a = Parameter(torch.randn(rank, in_features) * std_dev)
        self.b = Parameter(torch.zeros(out_features, rank))

        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        z = self.b @ self.a

        z *= self.alpha

        return x + z
