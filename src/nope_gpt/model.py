import copy

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
    Embedding,
    Linear,
    SiLU,
    RMSNorm,
    Dropout1d,
    Parameter,
)

from torch.nn.functional import softmax, scaled_dot_product_attention
from torch.nn.utils.parametrize import register_parametrization, remove_parametrizations
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from huggingface_hub import PyTorchModelHubMixin

from src.nope_gpt.caching import KVCache, DynamicKVBlock


class NoPEGPT(Module, PyTorchModelHubMixin):
    """A generative pretrained transformer with no positional embeddings."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimensions: int,
        num_q_heads: int,
        num_kv_heads: int,
        num_decoder_layers: int,
        hidden_ratio: int,
        dropout: float,
    ):
        super().__init__()

        assert vocabulary_size > 0, "Vocabulary size must be greater than 0."
        assert embedding_dimensions > 0, "Embedding dimensions must be greater than 0."

        assert (
            num_decoder_layers > 0
        ), "Number of decoder layers must be greater than 0."

        token_embeddings = Embedding(vocabulary_size, embedding_dimensions)

        decoder = Decoder(
            embedding_dimensions,
            num_q_heads,
            num_kv_heads,
            num_decoder_layers,
            hidden_ratio,
            dropout,
        )

        token_classifier = TokenClassifier(embedding_dimensions, vocabulary_size)

        token_classifier.linear.weight = token_embeddings.weight  # Tie weights

        self.token_embeddings = token_embeddings
        self.decoder = decoder
        self.token_classifier = token_classifier

    @property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def freeze_model_parameters(self) -> None:
        """Freeze all model parameters to prevent them from being updated during training."""

        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_token_embeddings(self) -> None:
        """Unfreeze the token embeddings to allow for fine-tuning."""

        self.token_embeddings.weight.requires_grad = True

    @torch.no_grad()
    def resize_token_embeddings(self, new_vocabulary_size: int) -> None:
        """Resize the token embeddings to a new vocabulary size."""

        vocabulary_size = self.token_embeddings.num_embeddings
        embedding_dimensions = self.token_embeddings.embedding_dim
        device = self.token_embeddings.weight.device

        new_embeddings = Embedding(new_vocabulary_size, embedding_dimensions).to(device)

        num_tokens_to_copy = min(new_vocabulary_size, vocabulary_size)

        new_embeddings.weight[:num_tokens_to_copy, :] = self.token_embeddings.weight[
            :num_tokens_to_copy, :
        ]

        # Initialize new embeddings with a kaiming normal distribution.
        for i in range(num_tokens_to_copy, new_vocabulary_size):
            tensor = torch.randn(embedding_dimensions)
            tensor /= sqrt(embedding_dimensions)

            new_embeddings.weight[i] = tensor

        self.token_embeddings.weight = new_embeddings.weight
        self.token_embeddings.num_embeddings = new_embeddings.num_embeddings

        # Retie weights
        self.token_classifier.linear.weight = self.token_embeddings.weight

    def add_lora_parameters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the model using LoRA adapters."""

        for module in self.decoder:
            module.add_lora_adapters(rank, alpha)

    def merge_lora_parameters(self) -> None:
        """Merge the LoRA parameters with the original parameters."""

        for module in self.modules():
            if hasattr(module, "parametrizations"):
                lora_params = [name for name in module.parametrizations.keys()]

                for name in lora_params:
                    remove_parametrizations(module, name)

    def forward(self, x: Tensor) -> Tensor:
        """A forward pass optimized for batch training."""

        z = self.token_embeddings.forward(x)
        z = self.decoder.forward(z)
        z = self.token_classifier.forward(z)

        return z

    @torch.no_grad()
    def predict(self, x: Tensor, kv_cache: KVCache) -> Tensor:
        """A forward pass optimized for next-token prediction."""

        z = self.token_embeddings.forward(x)
        z = self.decoder.predict(z, kv_cache)

        # Pluck only the last token embedding from each batch.
        z = z[:, -1, :]

        z = self.token_classifier.forward(z)

        return z

    @torch.no_grad()
    def generate(
        self,
        prompt: Tensor,
        max_tokens: int = 2000,
        context_length: int = 2048,
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

        assert max_tokens > 0, "Max tokens must be greater than 0."
        assert context_length > 0, "Context length must be greater than 0."
        assert temperature > 0, "Temperature must be greater than 0."
        assert 0.0 < top_p <= 1.0, "Top p must be between 0 and 1."
        assert 0.0 <= repeat_penalty <= 1.0, "Repeat penalty must be between 0 and 1."
        assert repeat_window > 0, "Repeat window must be greater than 0."

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


class Decoder(Module):
    """A stack of decoder blocks."""

    def __init__(
        self,
        embedding_dimensions: int,
        num_q_heads: int,
        num_kv_heads: int,
        num_layers: int,
        hidden_ratio: int,
        dropout: float,
    ):
        super().__init__()

        assert num_layers > 0, "Number of layers must be greater than 0."

        self.layers = ModuleList(
            [
                DecoderBlock(
                    embedding_dimensions,
                    num_q_heads,
                    num_kv_heads,
                    hidden_ratio,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.checkpoint = lambda layer, x: layer.forward(x)

    def enable_activation_checkpointing(self) -> None:
        """Instead of memorizing the activations of the forward pass, recompute them at various checkpoints."""

        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the decoder using LoRA adapters."""

        for layer in self.layers:
            layer.add_lora_adapters(rank, alpha)

    def forward(self, x: Tensor) -> Tensor:
        """A forward pass optimized for batch training."""

        for layer in self.layers:
            x = self.checkpoint(layer, x)

        return x

    @torch.no_grad()
    def predict(self, x: Tensor, kv_cache: KVCache) -> Tensor:
        """A forward pass optimized for next-token prediction."""

        for layer, kv_block in zip(self.layers, kv_cache):
            x = layer.predict(x, kv_block)

        return x


class DecoderBlock(Module):
    """Decoder block with multi-head attention, multilayer perceptron, and residual connections."""

    def __init__(
        self,
        embedding_dimensions: int,
        num_q_heads: int,
        num_kv_heads: int,
        hidden_ratio: int,
        dropout: float,
    ):
        super().__init__()

        self.stage1 = SelfAttention(
            embedding_dimensions, num_q_heads, num_kv_heads, dropout
        )

        self.stage2 = InvertedBottleneck(embedding_dimensions, hidden_ratio, dropout)

        self.norm1 = RMSNorm(embedding_dimensions)
        self.norm2 = RMSNorm(embedding_dimensions)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the decoder using LoRA adapters."""

        self.stage1.add_lora_adapters(rank, alpha)
        self.stage2.add_lora_adapters(rank, alpha)

    def forward(self, x: Tensor) -> Tensor:
        """A forward pass optimized for batch training."""

        z = self.norm1.forward(x)
        z = self.stage1.forward(z)

        x_hat = x + z  # Residual connection

        z = self.norm2.forward(x_hat)
        z = self.stage2.forward(z)

        z = x_hat + z  # Residual connection

        return z

    @torch.no_grad()
    def predict(self, x: Tensor, kv_block: DynamicKVBlock) -> Tensor:
        """A forward pass optimized for next-token prediction."""

        z = self.norm1.forward(x)
        z = self.stage1.predict(z, kv_block)

        x_hat = x + z  # Residual connection

        z = self.norm2.forward(x_hat)
        z = self.stage2.predict(z)

        z = x_hat + z  # Residual connection

        return z


class SelfAttention(Module):
    """Group query self-attention with causal masking for next-token prediction objective."""

    def __init__(
        self,
        embedding_dimensions: int,
        num_q_heads: int,
        num_kv_heads: int,
        dropout: float,
    ):
        super().__init__()

        assert embedding_dimensions > 0, "Embedding dimensions must be greater than 0."
        assert num_q_heads > 0, "Number of query heads must be greater than 0."
        assert num_kv_heads > 0, "Number of key-value heads must be greater than 0."

        assert (
            embedding_dimensions % num_q_heads == 0
        ), "Embedding dimensions must be divisible by the number of query heads."

        head_dimensions: int = embedding_dimensions // num_q_heads

        kv_dimensions: int = num_kv_heads * head_dimensions

        self.q_proj = Linear(embedding_dimensions, embedding_dimensions, bias=False)
        self.k_proj = Linear(embedding_dimensions, kv_dimensions, bias=False)
        self.v_proj = Linear(embedding_dimensions, kv_dimensions, bias=False)

        self.out_proj = Linear(embedding_dimensions, embedding_dimensions, bias=False)

        scale: float = 1.0 / sqrt(head_dimensions)

        is_gqa: bool = num_q_heads != num_kv_heads

        self.embedding_dimensions: int = embedding_dimensions
        self.num_q_heads: int = num_q_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dimensions: int = head_dimensions
        self.scale: float = scale
        self.is_gqa: bool = is_gqa
        self.dropout: float = dropout

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the attention module using LoRA adapters."""

        register_parametrization(
            self.q_proj, "weight", LoRA.from_linear(self.q_proj, rank, alpha)
        )

        register_parametrization(
            self.k_proj, "weight", LoRA.from_linear(self.k_proj, rank, alpha)
        )

        register_parametrization(
            self.v_proj, "weight", LoRA.from_linear(self.v_proj, rank, alpha)
        )

        register_parametrization(
            self.out_proj, "weight", LoRA.from_linear(self.out_proj, rank, alpha)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.size()

        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)

        q = q.view(b, t, self.num_q_heads, self.head_dimensions).transpose(1, 2)
        k = k.view(b, t, self.num_kv_heads, self.head_dimensions).transpose(1, 2)
        v = v.view(b, t, self.num_kv_heads, self.head_dimensions).transpose(1, 2)

        z = scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
            enable_gqa=self.is_gqa,
        )

        z = z.transpose(1, 2).contiguous().view(b, t, d)

        z = self.out_proj.forward(z)

        return z

    @torch.no_grad()
    def predict(self, x: Tensor, kv_block: DynamicKVBlock) -> Tensor:
        """A forward pass optimized for next-token prediction."""

        b, t, d = x.size()

        is_autoregressive_phase = t == 1

        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)

        q = q.view(b, t, self.num_q_heads, self.head_dimensions).transpose(1, 2)
        k = k.view(b, t, self.num_kv_heads, self.head_dimensions).transpose(1, 2)
        v = v.view(b, t, self.num_kv_heads, self.head_dimensions).transpose(1, 2)

        k, v = kv_block.update(k, v)

        z = scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            is_causal=not is_autoregressive_phase,
            enable_gqa=self.is_gqa,
        )

        z = z.transpose(1, 2).contiguous().view(b, t, d)

        z = self.out_proj.forward(z)

        return z


class InvertedBottleneck(Module):
    """A two layer fully-connected network with wide non-linear activations."""

    def __init__(self, embedding_dimensions: int, hidden_ratio: int, dropout: float):
        super().__init__()

        assert hidden_ratio in {1, 2, 4}, "Hidden ratio must be either 1, 2, or 4."

        hidden_dimensions: int = hidden_ratio * embedding_dimensions

        self.linear1 = Linear(embedding_dimensions, hidden_dimensions, bias=False)
        self.linear2 = Linear(hidden_dimensions, embedding_dimensions, bias=False)

        self.silu = SiLU()

        self.dropout = Dropout1d(p=dropout)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the feedforward module using LoRA adapters."""

        register_parametrization(
            self.linear1, "weight", LoRA.from_linear(self.linear1, rank, alpha)
        )

        register_parametrization(
            self.linear2, "weight", LoRA.from_linear(self.linear2, rank, alpha)
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.linear1.forward(x)
        z = self.silu.forward(z)
        z = self.dropout.forward(z)
        z = self.linear2.forward(z)

        return z

    def predict(self, x: Tensor) -> Tensor:
        z = self.linear1.forward(x)
        z = self.silu.forward(z)
        z = self.linear2.forward(z)

        return z


class TokenClassifier(Module):
    """A token classification head for the vocabulary."""

    def __init__(self, embedding_dimensions: int, vocabulary_size: int):
        super().__init__()

        self.norm = RMSNorm(embedding_dimensions)

        self.linear = Linear(embedding_dimensions, vocabulary_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm.forward(x)
        z = self.linear.forward(z)

        return z


class LoRA(Module):
    """Low rank weight decomposition transformation."""

    @classmethod
    def from_linear(cls, linear: Linear, rank: int, alpha: float) -> Self:
        out_features, in_features = linear.weight.shape

        return cls(in_features, out_features, rank, alpha)

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()

        assert rank > 0, "Rank must be greater than 0."
        assert alpha > 0.0, "Alpha must be greater than 0."

        lora_a = torch.randn(rank, in_features) / sqrt(rank)
        lora_b = torch.zeros(out_features, rank)

        self.lora_a = Parameter(lora_a)
        self.lora_b = Parameter(lora_b)

        self.alpha: float = alpha

    def forward(self, weight: Tensor) -> Tensor:
        z = self.lora_b @ self.lora_a

        z *= self.alpha

        z = weight + z

        return z
