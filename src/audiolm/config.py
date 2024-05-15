from pathlib import Path
from typing import Optional, Union
import torch

from pydantic import BaseModel, model_validator
from transformers import GPT2Tokenizer, GPT2Model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

gpt2_model = GPT2Model.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


class SemanticTransformerConfigure(BaseModel):
    vocab_size: int = 500
    context_length: int = 149
    embed_dim: int = gpt2_model.config.n_embd
    num_heads: int = gpt2_model.config.n_head
    layers: int = gpt2_model.config.n_layer
    feedforward_dim: int = gpt2_model.config.n_inner
    attn_dropout_prob: float = 0.1
    embed_dropout_prob: float = 0.1
    device: str = DEVICE

    @model_validator
    def check_embedding_size_num_heads_ratio(cls, values):
        num_heads = values.get("num_heads")
        embedding_size = values.get("embedding_size")
        if embedding_size % num_heads != 0:
            raise ValueError(f"{embedding_size=} is not divisible by {num_heads=}")
        return values

class AcousticTransformerConfigure(BaseModel):
    vocab_size: int = 0
    context_length: int = 0
    embed_dim: int = 0
    num_heads: int = 12
    layers: int = 12
    feedforward_dim: int = 3072
    attn_dropout_prob: float = 0.1
    embed_dropout_prob: float = 0.1
    device: str = DEVICE

    @model_validator
    def check_embedding_size_num_heads_ratio(cls, values):
        num_heads = values.get("num_heads")
        embedding_size = values.get("embedding_size")
        if embedding_size % num_heads != 0:
            raise ValueError(f"{embedding_size=} is not divisible by {num_heads=}")
        return values