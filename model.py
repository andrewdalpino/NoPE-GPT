from math import sqrt
from functools import partial
from typing import Self
from collections.abc import Generator
from collections import deque

import torch

from torch import Tensor
from torch.nn import (
    Module,
    ModuleList,
    Sequential,
    Embedding,
    Linear,
    SiLU,
    RMSNorm,
    Dropout1d,
    CrossEntropyLoss,
    Parameter,
)

from torch.nn.functional import softmax, scaled_dot_product_attention
from torch.nn.utils.parametrize import register_parametrization, remove_parametrizations
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from transformers import PretrainedConfig, PreTrainedModel

from caching import KVCache, DynamicKVBlock

from data import IGNORE_INDEX


class NoPEGPT(Module):
    """A generative pretrained transformer with no positional embeddings."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimensions: int,
        num_heads: int,
        num_layers: int,
        feed_forward_ratio: int,
        dropout: float,
    ):
        super().__init__()

        if vocabulary_size <= 0:
            raise ValueError(
                f"Vocabulary size must be greater than 0, {vocabulary_size} given."
            )

        if num_layers <= 0:
            raise ValueError(f"Num layers must be greater than 0, {num_layers} given.")

        token_embeddings = Embedding(vocabulary_size, embedding_dimensions)

        output_layer = Linear(embedding_dimensions, vocabulary_size, bias=False)

        output_layer.weight = token_embeddings.weight  # Tie weights

        self.token_embeddings = token_embeddings

        self.body = ModuleList(
            [
                DecoderBlock(
                    embedding_dimensions,
                    num_heads,
                    feed_forward_ratio,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.checkpoint = lambda layer, x: layer(x)

        self.output_norm = RMSNorm(embedding_dimensions)
        self.output_layer = output_layer

        self.loss_function = CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        self.vocabulary_size: int = vocabulary_size
        self.embedding_dimensions: int = embedding_dimensions
        self.num_heads: int = num_heads
        self.num_layers: int = num_layers

    @property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def enable_activation_checkpointing(self) -> None:
        """Instead of memorizing the activations of the forward pass, recompute them at various checkpoints."""

        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    def freeze_model_parameters(self) -> None:
        """Freeze all model parameters to prevent them from being updated during training."""

        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def resize_token_embeddings(self, vocabulary_size: int) -> None:
        """Resize the token embeddings to accommodate a new vocabulary size."""

        if vocabulary_size <= 0:
            raise ValueError(
                f"Vocabulary size must be greater than 0, {vocabulary_size} given."
            )

        new_embeddings = Embedding(vocabulary_size, self.embedding_dimensions)

        new_embeddings = new_embeddings.to(self.token_embeddings.weight.device)

        num_tokens_to_copy = min(vocabulary_size, self.token_embeddings.num_embeddings)

        new_embeddings.weight[:num_tokens_to_copy, :] = self.token_embeddings.weight[
            :num_tokens_to_copy, :
        ]

        # Initialize new embeddings with kaiming normal distribution.
        for i in range(num_tokens_to_copy, vocabulary_size):
            new_embeddings.weight[i] = torch.randn(self.embedding_dimensions) / sqrt(
                self.embedding_dimensions
            )

        self.token_embeddings.weight = new_embeddings.weight
        self.token_embeddings.num_embeddings = new_embeddings.num_embeddings

        self.output_layer.weight = self.token_embeddings.weight  # Retie weights

        self.vocabulary_size = vocabulary_size

    def unfreeze_token_embeddings(self) -> None:
        """Unfreeze the token embeddings to allow for fine-tuning."""

        self.token_embeddings.weight.requires_grad = True

    def add_lora_parameters(self, rank: int, alpha: float, dropout: float) -> None:
        """Reparameterize the weights of the model using LoRA adapters."""

        for module in self.body:
            register_parametrization(
                module.attention.qkv_proj,
                "weight",
                LoRA.from_linear(module.attention.qkv_proj, 3 * rank, alpha, dropout),
            )

            register_parametrization(
                module.attention.out_proj,
                "weight",
                LoRA.from_linear(module.attention.out_proj, rank, alpha, dropout),
            )

            register_parametrization(
                module.mlp.layers[0],
                "weight",
                LoRA.from_linear(module.mlp.layers[0], rank, alpha, dropout),
            )

            register_parametrization(
                module.mlp.layers[2],
                "weight",
                LoRA.from_linear(module.mlp.layers[2], rank, alpha, dropout),
            )

    def lora_state_dict(self) -> dict[str, Tensor]:
        """Return a state dict containing only the LoRA parameters."""

        return {
            name: module for name, module in self.state_dict().items() if "lora" in name
        }

    def merge_lora_parameters(self) -> None:
        """Merge the LoRA parameters with the original parameters."""

        for module in self.modules():
            if hasattr(module, "parametrizations"):
                lora_params = [name for name in module.parametrizations.keys()]

                for name in lora_params:
                    remove_parametrizations(module, name)

    def forward(
        self, x: Tensor, y: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """A forward pass optimized for batch training."""

        z = self.token_embeddings(x)

        for layer in self.body:
            z = self.checkpoint(layer, z)

        z = self.output_norm(z)
        z = self.output_layer(z)

        if y is not None:
            y_pred = z.view(-1, z.size(-1))
            labels = y.view(-1)  # Flatten the batch dimension.

            loss = self.loss_function(y_pred, labels)
        else:
            loss = None

        return z, loss

    @torch.no_grad()
    def predict(self, x: Tensor, kv_cache: KVCache) -> Tensor:
        """A forward pass optimized for next-token prediction."""

        z = self.token_embeddings(x)

        for layer, kv_block in zip(self.body, kv_cache):
            z = layer.predict(z, kv_block)

        z = z[:, -1, :]  # Pluck only the last token embedding from each batch.

        z = self.output_norm(z)
        z = self.output_layer(z)

        return z

    @torch.no_grad()
    def generate(
        self,
        prompt: Tensor,
        max_tokens: int = 1000,
        context_length: int = 1024,
        temperature: float = 1.0,
        top_k: int = 500,
        top_p: float = 0.9,
        repeat_penalty: float = 0.1,
        repeat_window: int = 50,
    ) -> Generator[tuple[Tensor, Tensor], None, int]:
        """
        Given a prompt, sample the next {max_tokens} tokens from the model weighted
        by their predicted probabilities and filtered by the {top_k} and {top_p}.
        """

        if max_tokens <= 0:
            raise ValueError(f"Max tokens must be greater than 0, {max_tokens} given.")

        if context_length <= 0:
            raise ValueError(
                f"Context length must be greater than 0, {context_length} given."
            )

        if temperature <= 0:
            raise ValueError(
                f"Temperature must be greater than 0, {temperature} given."
            )

        if top_k <= 0 or top_k > self.vocabulary_size:
            raise ValueError(
                f"Top k must be between 1 and {self.vocabulary_size}, {top_k} given."
            )

        if top_p <= 0.0 or top_p > 1.0:
            raise ValueError(f"Top p must be between 0 and 1, {top_p} given.")

        if repeat_penalty < 0.0 or repeat_penalty > 1.0:
            raise ValueError(
                f"Repeat penalty must be between 0 and 1, {repeat_penalty} given."
            )

        if repeat_window <= 0:
            raise ValueError(
                f"Repeat window must be greater than 0, {repeat_window} given."
            )

        kv_cache = KVCache(self, 1, context_length).to(prompt.device)

        prompt = prompt[-context_length:]

        previous_tokens = deque(maxlen=repeat_window)
        num_tokens = 0

        while num_tokens < max_tokens:
            logits = self.predict(prompt.unsqueeze(0), kv_cache).squeeze()

            for previous_token in previous_tokens:
                logits[previous_token] -= repeat_penalty * torch.abs(
                    logits[previous_token]
                )

            logits, indices = torch.topk(logits, top_k, sorted=True)

            logits /= temperature

            probabilities = softmax(logits, dim=0)

            cumulative_probability_mass = torch.cumsum(probabilities, dim=0)

            min_probability_mass = cumulative_probability_mass[0]

            threshold_p = max(top_p, min_probability_mass.item())

            selected_indices = cumulative_probability_mass <= threshold_p

            logits = logits[selected_indices]
            indices = indices[selected_indices]

            probabilities = softmax(logits, dim=0)

            offset = torch.multinomial(probabilities, num_samples=1).squeeze()

            next_token = indices[offset]
            probability = probabilities[offset]

            yield next_token, probability

            num_tokens += 1

            previous_tokens.append(next_token)

            prompt = next_token.unsqueeze(0)

        return num_tokens


class NoPEGPTHuggingFaceConfig(PretrainedConfig):
    """Provide a monolithic configuration object to enable compatibility with HuggingFace Transformers API."""

    model_type = "lightgpt"

    def __init__(
        self,
        vocabulary_size: int = 50257,
        embedding_dimensions: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        feed_forward_ratio: int = 4,
        dropout: float = 0.1,
        **kwargs,
    ):
        self.vocabulary_size = vocabulary_size
        self.embedding_dimensions = embedding_dimensions
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.feed_forward_ratio = feed_forward_ratio
        self.dropout = dropout

        super().__init__(**kwargs)


class NoPEGPTHuggingFaceModel(PreTrainedModel):
    """Wrap model to enable compatibility with HuggingFace Transformers API."""

    config_class = NoPEGPTHuggingFaceConfig

    def __init__(self, config: NoPEGPTHuggingFaceConfig):
        super().__init__(config)

        self.model = NoPEGPT(
            config.vocabulary_size,
            config.embedding_dimensions,
            config.num_heads,
            config.num_layers,
            config.feed_forward_ratio,
            config.dropout,
        )

    def forward(self, x: Tensor, y: Tensor | None = None) -> dict[str, Tensor | None]:
        logits, loss = self.model.forward(x, y)

        return {
            "logits": logits,
            "loss": loss,
        }


class DecoderBlock(Module):
    """Decoder block with multi-head attention, multilayer perceptron, and residual connections."""

    def __init__(
        self,
        embedding_dimensions: int,
        num_heads: int,
        feed_forward_ratio: int,
        dropout: float,
    ):
        super().__init__()

        self.norm1 = RMSNorm(embedding_dimensions)
        self.attention = SelfAttention(embedding_dimensions, num_heads, dropout)

        self.norm2 = RMSNorm(embedding_dimensions)
        self.mlp = MLP(embedding_dimensions, feed_forward_ratio, dropout)

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm1(x)
        z = self.attention(z)

        z = x + z  # Residual connection

        x = z

        z = self.norm2(x)
        z = self.mlp(z)

        z = x + z  # Residual connection

        return z

    @torch.no_grad()
    def predict(self, x: Tensor, kv_block: DynamicKVBlock) -> Tensor:
        """A forward pass optimized for next-token prediction."""

        z = self.norm1(x)
        z = self.attention.predict(z, kv_block)

        z = x + z  # Residual connection

        x = z

        z = self.norm2(x)
        z = self.mlp.predict(z)

        z = x + z  # Residual connection

        return z


class SelfAttention(Module):
    """Multihead self-attention with causal masking."""

    def __init__(self, embedding_dimensions: int, num_heads: int, dropout: float):
        super().__init__()

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

        self.qkv_proj = Linear(
            embedding_dimensions, 3 * embedding_dimensions, bias=False
        )

        self.out_proj = Linear(embedding_dimensions, embedding_dimensions, bias=False)

        head_dimensions: int = embedding_dimensions // num_heads
        scale: float = 1.0 / sqrt(head_dimensions)

        self.embedding_dimensions: int = embedding_dimensions
        self.num_heads: int = num_heads
        self.head_dimensions: int = head_dimensions
        self.scale: float = scale
        self.dropout: float = dropout

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.size()

        q, k, v = self.qkv_proj(x).split(self.embedding_dimensions, dim=-1)

        q = q.view(b, t, self.num_heads, self.head_dimensions).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dimensions).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dimensions).transpose(1, 2)

        z = scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )

        z = z.transpose(1, 2).contiguous().view(b, t, d)

        z = self.out_proj(z)

        return z

    @torch.no_grad()
    def predict(self, x: Tensor, kv_block: DynamicKVBlock) -> Tensor:
        """A forward pass optimized for next-token prediction."""

        b, t, d = x.size()

        is_autoregressive_phase = t == 1

        q, k, v = self.qkv_proj(x).split(self.embedding_dimensions, dim=-1)

        q = q.view(b, t, self.num_heads, self.head_dimensions).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dimensions).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dimensions).transpose(1, 2)

        k, v = kv_block.update(k, v)

        z = scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            is_causal=not is_autoregressive_phase,
        )

        z = z.transpose(1, 2).contiguous().view(b, t, d)

        z = self.out_proj(z)

        return z


class MLP(Module):
    """A two layer fully-connected network with dropout."""

    def __init__(
        self, embedding_dimensions: int, feed_forward_ratio: int, dropout: float
    ):
        super().__init__()

        if feed_forward_ratio not in {1, 2, 4}:
            raise ValueError("Feed-forward ratio must be either 1, 2, or 4.")

        hidden_dimensions: int = feed_forward_ratio * embedding_dimensions

        self.layers = Sequential(
            Linear(embedding_dimensions, hidden_dimensions, bias=False),
            SiLU(),
            Linear(hidden_dimensions, embedding_dimensions, bias=False),
        )

        self.dropout = Dropout1d(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.layers(x))

    def predict(self, x: Tensor) -> Tensor:
        return self.layers(x)


class LoRA(Module):
    """Low rank weight decomposition transformation."""

    @classmethod
    def from_linear(
        cls, linear: Linear, rank: int, alpha: float, dropout: float
    ) -> Self:
        out_features, in_features = linear.weight.shape

        return cls(in_features, out_features, rank, alpha, dropout)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError(f"Rank must be greater than 0, {rank} given.")

        if alpha <= 0.0:
            raise ValueError(f"Alpha must be greater than 0, {alpha} given.")

        lora_a = torch.randn(rank, in_features) / sqrt(rank)
        lora_b = torch.zeros(out_features, rank)

        self.lora_a = Parameter(lora_a)
        self.lora_b = Parameter(lora_b)

        self.dropout = Dropout1d(dropout)

        self.alpha: float = alpha

    def forward(self, weight: Tensor) -> Tensor:
        z = self.lora_b @ self.dropout(self.lora_a)

        z *= self.alpha

        return weight + z
