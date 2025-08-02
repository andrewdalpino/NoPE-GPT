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

from transformers import PretrainedConfig, PreTrainedModel

from caching import KVCache, DynamicKVBlock


class NoPEGPT(Module):
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

        if vocabulary_size <= 0:
            raise ValueError(
                f"Vocabulary size must be greater than 0, {vocabulary_size} given."
            )

        if num_decoder_layers <= 0:
            raise ValueError(f"Num decoder layers must be greater than 0.")

        token_embeddings = Embedding(vocabulary_size, embedding_dimensions)

        output_layer = Linear(embedding_dimensions, vocabulary_size, bias=False)

        output_layer.weight = token_embeddings.weight  # Tie weights

        self.token_embeddings = token_embeddings

        self.decoder = ModuleList(
            [
                DecoderBlock(
                    embedding_dimensions,
                    num_q_heads,
                    num_kv_heads,
                    hidden_ratio,
                    dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.checkpoint = lambda layer, x: layer.forward(x)

        self.output_norm = RMSNorm(embedding_dimensions)
        self.output_layer = output_layer

        self.vocabulary_size: int = vocabulary_size
        self.embedding_dimensions: int = embedding_dimensions
        self.num_q_heads: int = num_q_heads

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

    def unfreeze_last_k_decoder_layers(self, k: int) -> None:
        """Unfreeze the last k decoder layers to allow for fine-tuning."""

        if k <= 0:
            return
        
        for layer in self.decoder[-k:]:
            for param in layer.parameters():
                param.requires_grad = True

    def unfreeze_token_embeddings(self) -> None:
        """Unfreeze the token embeddings to allow for fine-tuning."""

        self.token_embeddings.weight.requires_grad = True

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

    def add_lora_parameters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the model using LoRA adapters."""

        for module in self.decoder:
            module.add_lora_adapters(rank, alpha)

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

    def forward(self, x: Tensor) -> Tensor:
        """A forward pass optimized for batch training."""

        z = self.token_embeddings.forward(x)

        for layer in self.decoder:
            z = self.checkpoint(layer, z)

        z = self.output_norm.forward(z)
        z = self.output_layer.forward(z)

        return z

    @torch.no_grad()
    def predict(self, x: Tensor, kv_cache: KVCache) -> Tensor:
        """A forward pass optimized for next-token prediction."""

        z = self.token_embeddings.forward(x)

        for layer, kv_block in zip(self.decoder, kv_cache):
            z = layer.predict(z, kv_block)

        # Pluck only the last token embedding from each batch.
        z = z[:, -1, :]

        z = self.output_norm.forward(z)
        z = self.output_layer.forward(z)

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

        assert max_tokens > 0, "Max tokens must be greater than 0."
        assert context_length > 0, "Context length must be greater than 0."
        assert temperature > 0, "Temperature must be greater than 0."

        assert (
            0 < top_k <= self.vocabulary_size
        ), "Top k must be between 1 and vocabulary size."

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


class NoPEGPTHuggingFaceConfig(PretrainedConfig):
    """Provide a monolithic configuration object to enable compatibility with HuggingFace Transformers API."""

    model_type = "nope-gpt"

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimensions: int,
        num_q_heads: int,
        num_kv_heads: int,
        num_decoder_layers: int,
        hidden_ratio: int,
        dropout: float,
        **kwargs,
    ):
        self.vocabulary_size = vocabulary_size
        self.embedding_dimensions = embedding_dimensions
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_decoder_layers = num_decoder_layers
        self.hidden_ratio = hidden_ratio
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
            config.num_q_heads,
            config.num_kv_heads,
            config.num_decoder_layers,
            config.hidden_ratio,
            config.dropout,
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        logits = self.model.forward(x)

        return {
            "logits": logits,
        }


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

        self.norm1 = RMSNorm(embedding_dimensions)
        self.attention = SelfAttention(
            embedding_dimensions, num_q_heads, num_kv_heads, dropout
        )

        self.norm2 = RMSNorm(embedding_dimensions)
        self.feedforward = InvertedBottleneck(
            embedding_dimensions, hidden_ratio, dropout
        )

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the decoder using LoRA adapters."""

        assert rank > 0, "Rank must be greater than 0."
        assert alpha > 0.0, "Alpha must be greater than 0."

        self.attention.add_lora_adapters(rank, alpha)
        self.feedforward.add_lora_adapters(rank, alpha)

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm1.forward(x)
        z = self.attention.forward(z)

        z = self.norm2.forward(z)
        z = self.feedforward.forward(z)

        z = x + z  # Residual connection

        return z

    @torch.no_grad()
    def predict(self, x: Tensor, kv_block: DynamicKVBlock) -> Tensor:
        """A forward pass optimized for next-token prediction."""

        z = self.norm1.forward(x)
        z = self.attention.predict(z, kv_block)

        z = self.norm2.forward(z)
        z = self.feedforward.predict(z)

        z = x + z  # Residual connection

        return z


class SelfAttention(Module):
    """Group query self-attention with causal masking."""

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

        is_gqa: bool = num_q_heads != num_kv_heads

        head_dimensions: int = embedding_dimensions // num_q_heads

        kv_dimensions: int = num_kv_heads * head_dimensions

        self.q_proj = Linear(embedding_dimensions, embedding_dimensions, bias=False)
        self.k_proj = Linear(embedding_dimensions, kv_dimensions, bias=False)
        self.v_proj = Linear(embedding_dimensions, kv_dimensions, bias=False)

        self.out_proj = Linear(embedding_dimensions, embedding_dimensions, bias=False)

        scale: float = 1.0 / sqrt(head_dimensions)

        self.embedding_dimensions: int = embedding_dimensions
        self.num_q_heads: int = num_q_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dimensions: int = head_dimensions
        self.is_gqa: bool = is_gqa
        self.scale: float = scale
        self.dropout: float = dropout

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the attention module using LoRA adapters."""

        assert rank > 0, "Rank must be greater than 0."
        assert alpha > 0.0, "Alpha must be greater than 0."

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

        assert rank > 0, "Rank must be greater than 0."
        assert alpha > 0.0, "Alpha must be greater than 0."

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

        return


class LoRA(Module):
    """Low rank weight decomposition transformation."""

    @classmethod
    def from_linear(cls, linear: Linear, rank: int, alpha: float) -> Self:
        out_features, in_features = linear.weight.shape

        return cls(in_features, out_features, rank, alpha)

    def __init__(
        self, in_features: int, out_features: int, rank: int, alpha: float
    ):
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
