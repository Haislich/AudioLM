"""# Jose: 
This module contains a custom definition of the Meta `EnCodec` model."""

from transformers import EncodecModel
from torch import nn, Tensor


class CustomEncodecModel(nn.Module):
    """This model is a wrapper around the `EnCodec` model
    pretrained and taken from the huggingface hub.
    It reshapes the output of the `EnCodec` quantizer to match Bert's output shape."""

    def __init__(self):
        """
        Initializes a `CustomEncodecModel` object.

        """
        super().__init__()
        self.encodec = EncodecModel.from_pretrained("facebook/encodec_24khz")

    def forward(self, x: Tensor, encode: bool = True):
        """
        Performs the forward pass of the CustomEncodec model.

        Args:
            x: torch.Tensor
                The input tensor.
            encode: bool
                flag indicating whether to encode or decode.

        Returns:
            If encode is True, returns the audio codes. Otherwise, returns the decoded output.
        """
        return (
            self.encodec.encode(x).audio_codes[0] if encode else self.encodec.decode(x)
        )
