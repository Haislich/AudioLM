import os

import torch
from torch import nn

from audiolm.absolute_transformer import TransformerDecoderOnly
from audiolm.custom_encodec import CustomEncodecModel
from audiolm.data_preparation import AudioDataLoader
from audiolm.w2v_hubert import W2VHuBERT_Quantizier


class AudioLM(nn.Module):
    def __init__(
        self,
        semantic_encoder: W2VHuBERT_Quantizier,
        semantic_transformer,
        acoustic_encoder: CustomEncodecModel,
        acoustic_transformer,
        fine_transformer,
        # https://stackoverflow.com/a/53797072
        *,
        audio_len=3,
        # We set Q' = 4 such that we predict the flattened tokens corresponding
        # to the coarse 4 layers in the second stage.
        n_coarse_quantizers=4,
        # Not specified, but num quantizers must be a power of 2
        # so this is the most reasonable combination.
        n_fine_quantizers=4,
    ) -> None:
        super().__init__()
        # TODO Freeze semantic encoder and decoder parameters.
        #
        # 'The tokenizer and detokenizer are pretrained and frozen ahead of training'.
        # Freezing the parameters for the acoustic token implicitly freezes the parameters
        # For the decoder
        #
        self.semantic_encoder = semantic_encoder
        for param in self.semantic_encoder.model.parameters():
            param.requires_grad = False
        self.acoustic_encoder = acoustic_encoder
        for param in self.acoustic_encoder.model.parameters():
            param.requires_grad = False
        self.semantic_transformer = semantic_transformer
        self.acoustic_transformer = acoustic_transformer
        self.fine_transformer = fine_transformer

    def forward(self, x: torch.Tensor):
        # region Semantic Modelling
        semantic_token = self.semantic_encoder(x)
        semantic_modelling = self.semantic_transformer(semantic_token)
        # endregion

        # region Coarse Acoustic Modelling
        coarse_acoustic_token, fine_acoustic_token, audio_scales = (
            self.acoustic_encoder.encode(x)
        )

        conditioning = torch.cat((semantic_modelling, coarse_acoustic_token), dim=1)
        coarse_acoustic_modelling = self.acoustic_transformer(
            conditioning.type(torch.int64)
        )
        # endregion

        # region Fine Acoustic modelling
        fine_acoustic_token = torch.Tensor(fine_acoustic_token)
        conditioning = torch.cat((coarse_acoustic_modelling, fine_acoustic_token))
        fine_acoustic_modelling = self.fine_transformer(conditioning, audio_scales)
        # endregion
        out = self.acoustic_encoder.decode(
            fine_acoustic_modelling,
        )
        return out


def pipeline():
    dataloader = AudioDataLoader(
        os.getcwd() + "\\data\\datasets\\", 2, max_length_audio=3
    )
    hubert = W2VHuBERT_Quantizier()
    semantic_transformer = TransformerDecoderOnly(
        500,  # Valore massimo del quantizzatore
        768,  # Numero arbitratio
    )
    encodec = CustomEncodecModel()
    acoustic_transfomer = TransformerDecoderOnly(
        1024,  # Valore massimo del quantizzatore
        1024,  # Numero arbitratio
    )
    fine_transformer = TransformerDecoderOnly(1024, 1024)
    return AudioLM(
        hubert, semantic_transformer, encodec, acoustic_transfomer, fine_transformer
    )(next(iter(dataloader)))


print(pipeline())
