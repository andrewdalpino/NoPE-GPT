import torch

from torch.nn import (
    Module,
    Sequential,
    Embedding,
    MultiheadAttention,
    Linear,
    LayerNorm,
    GELU,
    Dropout1d,
    CrossEntropyLoss,
)

from torch.nn.functional import softmax
from torch.nn.init import normal_


class GPT(Module):
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

        token_embeddings = Embedding(vocabulary_size, embedding_dimensions)
        positional_embeddings = Embedding(block_size, embedding_dimensions)

        output_layer = Linear(embedding_dimensions, vocabulary_size, bias=False)

        token_embeddings.weight = output_layer.weight  # Tie weights

        self.token_embeddings = token_embeddings
        self.positional_embeddings = positional_embeddings
        self.body = Sequential(
            *[
                CausalSelfAttentionBlock(
                    embedding_dimensions, block_size, num_heads, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = LayerNorm(embedding_dimensions, bias=False)
        self.output_layer = output_layer

        self.register_buffer("positions", torch.arange(0, block_size))

        self.loss_function = CrossEntropyLoss(ignore_index=-1)

        self._initialize()

        self.vocabulary_size = vocabulary_size
        self.block_size = block_size

    @property
    def num_trainable_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def _initialize(self):
        """Initialize the parameters of the network."""

        def init_weights(module):
            if isinstance(module, Embedding) or isinstance(module, Linear):
                normal_(module.weight, mean=0.0, std=0.02)

        self.apply(init_weights)

    def forward(self, x, y=None):
        b, t = x.size()

        pos = self.positions[:t]

        tok_out = self.token_embeddings(x)
        pos_out = self.positional_embeddings(pos)

        z = tok_out + pos_out

        z = self.body(z)

        z = self.output_norm(z)
        z = self.output_layer(z)

        if y is not None:
            loss = self.loss_function(z.view(-1, z.size(-1)), y.view(-1))

        else:
            loss = None

        return z, loss

    @torch.no_grad()
    def generate(self, prompts, max_tokens: int, temperature: float, top_k: int):
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

        out = prompts

        for i in range(max_tokens):
            context_window = out[:, -self.block_size :]

            y_pred, _ = self.forward(context_window)

            y_pred = y_pred[:, -1, :]

            values, indices = torch.topk(y_pred, top_k, sorted=False)

            values /= temperature

            probabilities = softmax(values, dim=-1)

            offsets = torch.multinomial(probabilities, num_samples=1).squeeze(0)

            next_tokens = indices.index_select(-1, offsets)

            out = torch.cat((out, next_tokens), dim=1)

        return out


class CausalSelfAttentionBlock(Module):
    """Causal self-attention block with additional residual connections."""

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

        causal_mask = torch.tril(torch.ones((block_size, block_size)))
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float("-Inf"))
        causal_mask = causal_mask.masked_fill(causal_mask == 1, 0.0)

        self.register_buffer("mask", causal_mask)

    def forward(self, x):
        b, t, c = x.size()

        causal_mask = self.mask[:t, :t]

        z = self.norm1(x)
        z, _ = self.attention(z, z, z, is_causal=True, attn_mask=causal_mask)

        z = x + z  # Residual connection

        x = z

        z = self.norm2(x)
        z = self.mlp(z)

        z = x + z  # Residual connection

        return z


class MLP(Module):
    def __init__(
        self, embedding_dimensions: int, hidden_dimensions: int, dropout: float
    ):
        super().__init__()

        if embedding_dimensions <= 0:
            raise ValueError(
                f"Embedding dimensions must be greater than 0, {embedding_dimensions} given."
            )

        self.layers = Sequential(
            Linear(embedding_dimensions, hidden_dimensions, bias=False),
            GELU(),
            Linear(hidden_dimensions, embedding_dimensions, bias=False),
            Dropout1d(dropout),
        )

    def forward(self, x):
        return self.layers.forward(x)
