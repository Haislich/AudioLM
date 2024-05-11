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

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass of the CustomEncodec model using the base Encoder.

        """
        return self.encodec(x)

    def encode(self, x: Tensor) -> Tensor:
        """Encode the input into a discrete representation."""
        return self.encodec.encode(x)


if __name__ == "__main__":
    import os
    from audiolm.data_preparation import AudioDataLoader

    dataloader = AudioDataLoader(os.getcwd() + "\\data\\datasets\\mini")
    encodec = CustomEncodecModel()
    print(encodec.encode(next(iter(dataloader))))
