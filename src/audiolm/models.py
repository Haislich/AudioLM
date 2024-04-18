from torch import nn


class TokenizerModel(nn.Module):
    """The tokenizer model maps an input `x` of length `T` into an encoded sequence `h` of length `T'`."""


class DecoderOnly(nn.Module):
    """The decoder-only transformer language model, operates on discrete tokens `y`."""


class DetokenizerModel(nn.Module):
    """The detokenizer model maps the sequence of predicted tokens back to audio"""
