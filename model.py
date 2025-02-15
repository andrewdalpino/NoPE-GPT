from math import sqrt
from dataclasses import dataclass
from functools import partial, cached_property
from typing import Iterator, Self

import torch

from torch import Tensor
from torch.nn import (
    Module,
    ModuleList,
    Sequential,
    Embedding,
    MultiheadAttention,
    Linear,
    SiLU,
    RMSNorm,
    Dropout1d,
    CrossEntropyLoss,
    Parameter,
)

from torch.nn.functional import softmax, log_softmax
from torch.nn.utils.parametrize import register_parametrization, remove_parametrizations
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from huggingface_hub import PyTorchModelHubMixin


class LightGPT(Module, PyTorchModelHubMixin):
    """A generative pretrained transformer."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimensions: int,
        num_heads: int,
        num_layers: int,
        feed_forward_ratio: int,
        dropout: float,
        padding_index: int,
        eos_index: int,
    ):
        super().__init__()

        if num_layers <= 0:
            raise ValueError(f"Num layers must be greater than 0, {num_layers} given.")

        if feed_forward_ratio not in {1, 2, 4}:
            raise ValueError("Feed-forward ratio must be either 1, 2, or 4.")

        if vocabulary_size <= 0:
            raise ValueError(
                f"Vocabulary size must be greater than 0, {vocabulary_size} given."
            )

        token_embeddings = Embedding(
            vocabulary_size, embedding_dimensions, padding_idx=padding_index
        )

        output_layer = Linear(embedding_dimensions, vocabulary_size, bias=False)

        token_embeddings.weight = output_layer.weight  # Tie weights

        self.token_embeddings = token_embeddings

        self.body = ModuleList(
            [
                CausalSelfAttentionBlock(
                    embedding_dimensions,
                    num_heads,
                    feed_forward_ratio,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.checkpoint = lambda layer, x, attention_mask: layer(x, attention_mask)

        self.output_norm = RMSNorm(embedding_dimensions)
        self.output_layer = output_layer

        self.loss_function = CrossEntropyLoss(ignore_index=padding_index)

        self.vocabulary_size = vocabulary_size
        self.eos_index = eos_index

    @cached_property
    def num_trainable_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def enable_activation_checkpointing(self) -> None:
        """Instead of memorizing the activations of the forward pass, recompute them at various checkpoints."""
        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    def forward(
        self, x: Tensor, y: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """A forward pass optimized for batch training."""

        z = self.token_embeddings(x)

        b, t, d = z.size()

        causal_mask = torch.full((t, t), float("-inf"), dtype=z.dtype, device=z.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)

        for layer in self.body:
            z = self.checkpoint(layer, z, causal_mask)

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
    def predict(self, x: Tensor) -> Tensor:
        """A forward pass optimized for batch next-token prediction."""

        z = self.token_embeddings(x)

        b, t, d = z.size()

        causal_mask = torch.full((t, t), float("-inf"), dtype=z.dtype, device=z.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)

        for layer in self.body:
            z = layer(z, causal_mask)

        z = self.output_norm(z)

        z = z[:, -1, :]  # Pluck only the last token embedding from each batch.

        z = self.output_layer(z)

        return z

    @torch.no_grad()
    def generate(
        self,
        prompt: Tensor,
        max_tokens: int = 2000,
        context_length: int = 1024,
        temperature: float = 1.0,
        top_k: int = 500,
        top_p: float = 0.9,
    ) -> Iterator:
        """
        Given a prompt, sample the next {max_tokens} tokens from the model weighted
        by their predicted probabilities and filtered by the {top_k} and {top_p}.
        """

        if max_tokens <= 0:
            raise ValueError(f"Max tokens must be greater than 0, {max_tokens} given.")

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

        context_window = prompt

        for _ in range(max_tokens):
            context_window = context_window[-context_length:]

            logits = self.predict(context_window.unsqueeze(0)).squeeze()

            logits, indices = torch.topk(logits, top_k, sorted=True)

            probabilities = softmax(logits, dim=0)

            cumulative_probability_mass = torch.cumsum(probabilities, dim=0)

            min_probability_mass = cumulative_probability_mass[0]

            threshold_p = max(top_p, min_probability_mass.item())

            selected_indices = cumulative_probability_mass <= threshold_p

            logits = logits[selected_indices]
            indices = indices[selected_indices]

            logits /= temperature

            probabilities = softmax(logits, dim=0)

            offset = torch.multinomial(probabilities, num_samples=1).squeeze()

            next_token = indices[offset]

            if next_token == self.eos_index:
                break

            yield next_token

            context_window = torch.cat((context_window, next_token.unsqueeze(0)))

    @torch.no_grad()
    def beam_search(
        self,
        prompt: Tensor,
        max_tokens: int = 100,
        context_length: int = 1024,
        num_candidates: int = 3,
        beam_width: int = 16,
    ) -> list:
        """
        Given a prompt, return the {num_candidates} highest probability sequences. Note that
        this method is often best for generating shorter sequences and is typically less
        natural sounding than sequences that are more random in nature.
        """

        if max_tokens <= 0:
            raise ValueError(f"Max tokens must be greater than 0, {max_tokens} given.")

        if num_candidates <= 0:
            raise ValueError(
                f"Num candidates must be greater than 0, {num_candidates} given."
            )

        if beam_width <= 0:
            raise ValueError(f"Beam width must be greater than 0, {beam_width} given.")

        @dataclass
        class Candidate:
            log_probability: float
            tokens: Tensor

            def priority(self) -> float:
                return self.log_probability

        sort_candidates = partial(
            sorted,
            key=lambda candidate: candidate.priority(),
            reverse=True,
        )

        candidates: list[Candidate] = []
        completed: list[Candidate] = []

        tokens = torch.tensor([], dtype=prompt.dtype).to(prompt.device)

        candidates.append(Candidate(0.0, tokens))

        while len(candidates) > 0:
            candidate = candidates.pop()

            if len(completed) >= num_candidates:
                completed = sort_candidates(completed)

                completed = completed[:num_candidates]

                worst_candidate = completed[-1]

                if candidate.log_probability < worst_candidate.log_probability:
                    break

            if len(candidate.tokens) > 0 and candidate.tokens[-1] == self.eos_index:
                candidate.tokens = candidate.tokens[:-1]

                completed.append(candidate)

                continue

            if len(candidate.tokens) >= max_tokens:
                completed.append(candidate)

                continue

            context_window = torch.cat((prompt, candidate.tokens))

            context_window = context_window[-context_length:]

            logits = self.predict(context_window.unsqueeze(0)).squeeze()

            logits, indices = torch.topk(logits, beam_width, sorted=False)

            log_probabilities = log_softmax(logits, dim=0)

            for log_probability, index in zip(log_probabilities, indices):
                log_probability = candidate.log_probability + log_probability

                tokens = torch.cat((candidate.tokens, index.unsqueeze(0)))

                candidates.append(Candidate(log_probability, tokens))

            candidates = sort_candidates(candidates)

            candidates = candidates[:beam_width]

        return completed


class LightGPTInstruct(Module, PyTorchModelHubMixin):
    """
    A wrapper for pretrained GPT models that applies a LoRA reparameterization
    to the intermediate layers of the network.
    """

    def __init__(self, model: LightGPT, rank: int, alpha: float, dropout: float):
        super().__init__()

        if rank <= 0:
            raise ValueError(f"Rank must be greater than 0, {rank} given.")

        if alpha <= 0.0:
            raise ValueError(f"Alpha must be greater than 0, {alpha} given.")

        for param in model.parameters():
            param.requires_grad = False

        model.output_layer.weight = Parameter(
            torch.cat(
                (
                    model.output_layer.weight,
                    torch.randn(2, model.output_layer.weight.size(dim=1)),
                )
            )
        )

        model.token_embeddings.weight = model.output_layer.weight

        for module in model.body:
            out_features, in_features = module.attention.in_proj_weight.shape

            register_parametrization(
                module.attention,
                "in_proj_weight",
                LoRA(in_features, out_features, rank, alpha, dropout),
            )

            out_features, in_features = module.attention.out_proj.weight.shape

            register_parametrization(
                module.attention.out_proj,
                "weight",
                LoRA(in_features, out_features, rank, alpha, dropout),
            )

            for layer in module.mlp.layers:
                if isinstance(layer, Linear):
                    register_parametrization(
                        layer,
                        "weight",
                        LoRA.from_linear(layer, rank, alpha, dropout),
                    )

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

    def merge_lora_parameters(self):
        """Merge the LoRA parameters with the original parameters."""

        for module in self.model.modules():
            if hasattr(module, "parametrizations"):
                lora_params = [name for name in module.parametrizations.keys()]

                for name in lora_params:
                    remove_parametrizations(module, name, leave_parametrized=True)

    def forward(
        self, x: Tensor, y: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        return self.model.forward(x, y)

    def predict(self, x: Tensor) -> Tensor:
        return self.model.predict(x)

    def generate(
        self,
        prompt: Tensor,
        max_tokens: int = 2000,
        context_length: int = 1024,
        temperature: float = 1.0,
        top_k: int = 500,
        top_p: float = 0.9,
    ) -> Iterator:
        return self.model.generate(
            prompt, max_tokens, context_length, temperature, top_k, top_p
        )

    def beam_search(
        self,
        prompt: Tensor,
        max_tokens: int = 100,
        context_length: int = 1024,
        num_candidates: int = 3,
        beam_width: int = 16,
    ) -> list:
        return self.model.beam_search(
            prompt, max_tokens, context_length, num_candidates, beam_width
        )


class ONNXModel(Module):
    """This wrapper provides a clean inferencing API for ONNX production models."""

    def __init__(self, model: LightGPT | LightGPTInstruct):
        super().__init__()

        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model.predict(x)


class CausalSelfAttentionBlock(Module):
    """Causal self-attention block with residual connections."""

    def __init__(
        self,
        embedding_dimensions: int,
        num_heads: int,
        feed_forward_ratio: int,
        dropout: float,
    ):
        super().__init__()

        if embedding_dimensions <= 0:
            raise ValueError(
                f"Embedding dimensions must be greater than 0, {embedding_dimensions} given."
            )

        if num_heads <= 0:
            raise ValueError(f"Num heads must be greater than 0, {num_heads} given.")

        if dropout < 0 or dropout > 1:
            raise ValueError(f"Dropout must be between 0 and 1, {dropout} given")

        self.norm1 = RMSNorm(embedding_dimensions)
        self.attention = MultiheadAttention(
            embedding_dimensions,
            num_heads,
            batch_first=True,
            dropout=dropout,
            bias=False,
        )

        hidden_dimensions = feed_forward_ratio * embedding_dimensions

        self.norm2 = RMSNorm(embedding_dimensions)
        self.mlp = MLP(embedding_dimensions, hidden_dimensions, dropout)

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        z = self.norm1(x)
        z, _ = self.attention(z, z, z, attn_mask=attention_mask, is_causal=True)

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
            SiLU(),
            Linear(hidden_dimensions, embedding_dimensions, bias=False),
        )

        self.dropout = Dropout1d(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.layers(x))


class LoRA(Module):
    """Rank decomposition transformation."""

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

        std_dev = 1.0 / sqrt(rank)

        self.lora_a = Parameter(torch.randn(rank, in_features) * std_dev)
        self.lora_b = Parameter(torch.zeros(out_features, rank))

        self.dropout = Dropout1d(p=dropout)

        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        z = self.lora_b @ self.dropout(self.lora_a)

        z *= self.alpha

        return x + z
