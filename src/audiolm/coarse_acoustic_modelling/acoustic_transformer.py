from typing import Any, Callable
from torch import Tensor, nn
from torch.nn.functional import relu


class AcousticTransformer(nn.Transformer):
    """We use identical decoder-only Transformers in
    all stages, with 12 layers, 16 attention heads, embedding
    dimension of 1024, feed-forward layer dimension of 4096
    and dropout of 0.1, together with T5-style relative positional
    embeddings [38], resulting in a model parameter size of
    0.3B per stage. During training, we use random cropping
    to equivalent input lengths of 30, 10 and 3 seconds for the
    three stages. Furthermore, in the first two stages, we follow
    the previously proposed practice of removing consecutive
    repetitions of the semantic tokens [14]. We train each stage
    on 16 TPUv4s with batch size of 256 for 1M steps."""

    def __init__(
        self,
        d_model: int = 1024,  # embedding dimension of 1024
        nhead: int = 16,  # 16 attention heads
        num_encoder_layers: int = 12,  # 12 layers
        num_decoder_layers: int = 12,  # 12 layers
        dim_feedforward: int = 4096,  # feed-forward layer dimension of 4096
        dropout: float = 0.1,  # dropout of 0.1
        activation: str | Callable[[Tensor], Tensor] = relu,
        custom_encoder: Any | None = None,
        custom_decoder: Any | None = None,
        layer_norm_eps: float = 0.00001,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            activation,
            custom_encoder,
            custom_decoder,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            device,
            dtype,
        )

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        src_is_causal: bool | None = None,
        tgt_is_causal: bool | None = None,
        memory_is_causal: bool = False,
    ) -> Tensor:
        return super().forward(
            src,
            tgt,
            src_mask,
            tgt_mask,
            memory_mask,
            src_key_padding_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            src_is_causal,
            tgt_is_causal,
            memory_is_causal,
        )
