from math import sqrt
from functools import partial
from typing import Self
from collections import deque
from collections.abc import Generator

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

from torch.nn.functional import scaled_dot_product_attention, softmax, log_softmax
from torch.nn.utils.parametrize import register_parametrization, remove_parametrizations
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from huggingface_hub import PyTorchModelHubMixin

from src.nope_gpt.caching import KVCache, DynamicKVBlock
from src.nope_gpt.search import Candidate


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

        token_classifier.tie_weights(token_embeddings)

        self.vocabulary_size = vocabulary_size
        self.embedding_dimensions = embedding_dimensions
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

        new_embeddings = Embedding(new_vocabulary_size, self.embedding_dimensions)

        new_embeddings.to(self.token_embeddings.weight.device)

        num_tokens_to_copy = min(new_vocabulary_size, self.vocabulary_size)

        new_embeddings.weight[:num_tokens_to_copy, :] = self.token_embeddings.weight[
            :num_tokens_to_copy, :
        ]

        # Initialize new embeddings with a kaiming normal distribution.
        for i in range(num_tokens_to_copy, new_vocabulary_size):
            tensor = torch.randn(self.embedding_dimensions)

            tensor /= sqrt(self.embedding_dimensions)

            new_embeddings.weight[i] = tensor

        # Retie weights
        self.token_classifier.tie_weights(new_embeddings)

        self.vocabulary_size = new_vocabulary_size
        self.token_embeddings = new_embeddings

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the model using LoRA adapters."""

        for module in self.decoder.layers:
            module.add_lora_adapters(rank, alpha)

    def merge_lora_adapters(self) -> None:
        """Merge the LoRA adapters with the original parameters."""

        for module in self.modules():
            if not hasattr(module, "parametrizations"):
                continue

            lora_params = []

            for name, parameterizations in module.parametrizations.items():
                for parametrization in parameterizations:
                    if isinstance(parametrization, LoRA):
                        lora_params.append(name)

            for name in lora_params:
                remove_parametrizations(module, name)

    def forward(self, x: Tensor) -> Tensor:
        """A forward pass optimized for batch training."""

        z = self.token_embeddings.forward(x)
        z = self.decoder.forward(z)
        z = self.token_classifier.forward(z)

        return z

    def predict(self, x: Tensor, kv_cache: KVCache) -> Tensor:
        """A forward pass optimized for autoregressive next-token prediction."""

        z = self.token_embeddings.forward(x)
        z = self.decoder.predict(z, kv_cache)

        # Pluck only the last token embedding from each batch.
        z = z[:, -1, :]

        z = self.token_classifier.forward(z)

        return z

    @torch.inference_mode()
    def generate(
        self,
        prompt: Tensor,
        max_tokens: int = 2000,
        context_length: int = 4096,
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

        kv_cache = KVCache(self.decoder, 1, context_length).to(prompt.device)

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

    @torch.inference_mode()
    def beam_search(
        self,
        prompt: Tensor,
        max_tokens: int = 2000,
        context_length: int = 4096,
        num_candidates: int = 3,
        beam_width: int = 16,
        length_penalty: float = 1.0,
        eos_indices: set = set(),
    ) -> list[Candidate]:
        """
        Given a prompt, return the {num_candidates} highest probability sequences.
        """

        assert max_tokens > 0, "Max tokens must be greater than 0."
        assert context_length > 0, "Context length must be greater than 0."
        assert num_candidates > 0, "Num candidates must be greater than 0."
        assert beam_width > 0, "Beam width must be greater than 0."
        assert length_penalty > 0, "Length penalty must be greater than 0."

        new_candidate = partial(Candidate, length_penalty=length_penalty)

        sort_candidates = partial(
            sorted,
            key=lambda candidate: candidate.priority,
            reverse=True,
        )

        candidates: list[Candidate] = []
        completed: list[Candidate] = []

        tokens = torch.empty(0, dtype=prompt.dtype).to(prompt.device)

        candidates.append(new_candidate(0.0, tokens))

        while len(candidates) > 0:
            candidate = candidates.pop()

            if len(completed) >= num_candidates:
                completed = sort_candidates(completed)

                completed = completed[:num_candidates]

                worst_candidate = completed[-1]

                if (
                    candidate.cumulative_log_probability
                    < worst_candidate.cumulative_log_probability
                ):
                    break

            if len(candidate.tokens) > 0:
                last_token = candidate.tokens[-1]

                if last_token.item() in eos_indices:
                    candidate.tokens = candidate.tokens[:-1]

                    completed.append(candidate)

                    continue

            if len(candidate.tokens) >= max_tokens:
                completed.append(candidate)

                continue

            context_window = torch.cat((prompt, candidate.tokens))

            context_window = context_window[-context_length:]

            logits = self.forward(context_window.unsqueeze(0)).squeeze()

            logits = logits[-1]

            logits, indices = torch.topk(logits, beam_width, sorted=False)

            log_probabilities = log_softmax(logits, dim=0)

            for log_probability, index in zip(log_probabilities, indices):
                cumulative_log_probability = (
                    candidate.cumulative_log_probability + log_probability
                )

                tokens = torch.cat((candidate.tokens, index.unsqueeze(0)))

                candidates.append(new_candidate(cumulative_log_probability, tokens))

            candidates = sort_candidates(candidates)

            candidates = candidates[:beam_width]

        return completed


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
        for layer in self.layers:
            x = self.checkpoint(layer, x)

        return x

    def predict(self, x: Tensor, kv_cache: KVCache) -> Tensor:
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
        z = self.norm1.forward(x)
        z = self.stage1.forward(z)

        x_hat = x + z  # Residual connection

        z = self.norm2.forward(x_hat)
        z = self.stage2.forward(z)

        z = x_hat + z  # Residual connection

        return z

    def predict(self, x: Tensor, kv_block: DynamicKVBlock) -> Tensor:
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
            num_q_heads >= num_kv_heads
        ), "Number of query heads must be greater than or equal to the number of key-value heads."

        assert (
            embedding_dimensions % num_q_heads == 0
        ), "Embedding dimensions must be divisible by the number of query heads."

        head_dimensions = embedding_dimensions // num_q_heads

        kv_dimensions = num_kv_heads * head_dimensions

        self.q_proj = Linear(embedding_dimensions, embedding_dimensions, bias=False)
        self.k_proj = Linear(embedding_dimensions, kv_dimensions, bias=False)
        self.v_proj = Linear(embedding_dimensions, kv_dimensions, bias=False)

        self.out_proj = Linear(embedding_dimensions, embedding_dimensions, bias=False)

        scale = 1.0 / sqrt(head_dimensions)

        is_gqa = num_q_heads > num_kv_heads

        self.embedding_dimensions = embedding_dimensions
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dimensions = head_dimensions
        self.scale = scale
        self.is_gqa = is_gqa
        self.dropout = dropout

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

    def predict(self, x: Tensor, kv_block: DynamicKVBlock) -> Tensor:
        b, t, d = x.size()

        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)

        q = q.view(b, t, self.num_q_heads, self.head_dimensions).transpose(1, 2)
        k = k.view(b, t, self.num_kv_heads, self.head_dimensions).transpose(1, 2)
        v = v.view(b, t, self.num_kv_heads, self.head_dimensions).transpose(1, 2)

        k, v = kv_block.update(k, v)

        is_autoregressive_phase = t == 1

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

    def tie_weights(self, token_embeddings: Embedding) -> None:
        """Tie the weights of the token classifier to the token embeddings."""

        assert (
            self.linear.weight.shape == token_embeddings.weight.shape
        ), "Token embeddings must have the same shape as token classifier."

        self.linear.weight = token_embeddings.weight

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

        self.alpha = alpha

    def forward(self, weight: Tensor) -> Tensor:
        z = self.lora_b @ self.lora_a

        z *= self.alpha

        z = weight + z

        return z
