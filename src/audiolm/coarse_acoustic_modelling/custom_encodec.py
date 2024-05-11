"""# Jose: 
This module contains a custom definition of the Meta `EnCodec` model."""

from typing import Optional
import torch
from torch import nn
from transformers import EncodecModel


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the CustomEncodec model using the base Encoder.

        """
        return self.encodec(x)

    # https://github.com/huggingface/transformers/blob/e0c3cee17085914bbe505c159beeb8ae39bc37dd/src/transformers/models/encodec/modeling_encodec.py#L580
    def encode(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None,
        bandwidth: Optional[float] = None,
    ) -> torch.Tensor:
        """Encode the input into a discrete representation."""
        return self.encodec.encode(x, padding_mask, bandwidth)


if __name__ == "__main__":
    import os
    from audiolm.data_preparation import AudioDataLoader

    dataloader = AudioDataLoader(
        data_path=os.getcwd() + "\\data\\datasets\\mini",
        batch_size=2,
        shuffle=False,
        max_length_audio=20,
        sample_frequency=24000,
    )
    print(next(iter(dataloader)))
