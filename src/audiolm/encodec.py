"""This module contains a custom definition of the Meta `EnCodec` model."""

from typing import Optional, Tuple
import torch
from torch import nn
from transformers import EncodecModel
import torchaudio


class Encodec(nn.Module):
    """This model is a wrapper around the `EnCodec` model
    pretrained and taken from the huggingface hub.
    It reshapes the output of the `EnCodec` quantizer to match Bert's output shape."""

    def __init__(self):
        """
        Initializes a `Encodec` object.

        """
        super().__init__()
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz")

    def forward(self, *_, **__) -> torch.Tensor:
        """
        Performs the forward pass of the CustomEncodec model using the base Encoder.

        """
        return NotImplementedError

    # https://github.com/huggingface/transformers/blob/e0c3cee17085914bbe505c159beeb8ae39bc37dd/src/transformers/models/encodec/modeling_encodec.py#L580
    def encode(
        self,
        input_values: torch.Tensor,
        in_sample_freq=16000,
        # Number of coarse quantizers
        coarse_quantizers=4,
    ) -> Tuple[list, list, list]:
        """Encode the input into a discrete representation."""
        input_values = torchaudio.functional.resample(
            input_values, in_sample_freq, self.model.config.sampling_rate
        )
        coarse = []
        fine = []
        audio_scales = []
        with torch.no_grad():
            for batch in input_values:
                # https://github.com/facebookresearch/encodec/tree/main#extracting-discrete-representations
                # TODO: Understand the mapping of the codebooks
                encoded_frames = self.model.encode(batch.unsqueeze(0), bandwidth=6)
                # Remove the useless dimensions
                codes = encoded_frames.audio_codes.squeeze((0, 1))

                num_quantizers, num_tokens = codes.shape
                fine_quantizers = num_quantizers - coarse_quantizers
                coarse_codes, fine_codes = codes.split(
                    [coarse_quantizers, fine_quantizers], dim=0
                )
                # Return the codes in row-major order

                coarse_codes = coarse_codes.T.reshape(coarse_quantizers * num_tokens)
                coarse.append(coarse_codes)
                fine_codes = fine_codes.T.reshape(fine_quantizers * num_tokens)
                fine.append(fine_codes)
                audio_scales.append(encoded_frames.audio_scales)
        return (torch.stack(coarse), torch.stack(fine), audio_scales)

    # https://github.com/huggingface/transformers/blob/e0c3cee17085914bbe505c159beeb8ae39bc37dd/src/transformers/models/encodec/modeling_encodec.py#L708
    def decode(
        self,
        audio_codes: torch.Tensor,
        audio_scales: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """Decodes the given frames into an output audio waveform."""
        return self.model.decode(audio_codes, audio_scales, padding_mask, None)
