import os

import torch
from torch import nn

from audiolm.absolute_transformer import (
    SemanticTransformer,
    CoarseAcousticTransformer,
    FineAcousticTransformer,
)
from audiolm.custom_encodec import CustomEncodecModel
from audiolm.data_preparation import AudioDataLoader
from audiolm.w2v_hubert import W2VHuBERT_Quantizier


class AudioLM:
    def __init__(
        self,
        semantic_encoder: W2VHuBERT_Quantizier,
        semantic_transformer: SemanticTransformer,
        acoustic_encoder_decoder: CustomEncodecModel,
        coarse_acoustic_transformer: CoarseAcousticTransformer,
        fine_acoustic_transformer: FineAcousticTransformer,
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
        self.acoustic_encoder_decoder = acoustic_encoder_decoder
        for param in self.acoustic_encoder_decoder.model.parameters():
            param.requires_grad = False
        self.semantic_transformer = semantic_transformer
        self.coarse_acoustic_transformer = coarse_acoustic_transformer
        self.fine_acoustic_transformer = fine_acoustic_transformer
        self.audio_len = audio_len
        self.n_coarse_quantizers = n_coarse_quantizers
        self.n_fine_quantizers = n_fine_quantizers

    def inference(self, x: torch.Tensor):
        # Add dimension at the beginning to simulate being a batch
        # to conform with the rest of the API
        x = x.unsqueeze(0)
        semantic_encode = self.semantic_encoder(x)
        semantic_token = self.semantic_transformer.generate(
            semantic_encode, self.audio_len
        )

        coarse_acoustic_tokens, fine_acoustic_tokens, _ = (
            self.acoustic_encoder_decoder.encode(x, self.n_coarse_quantizers)
        )
        coarse_conditioning = torch.cat((semantic_token, coarse_acoustic_tokens), dim=1)
        coarse_tokens = self.coarse_acoustic_transformer.generate(
            coarse_conditioning, 3
        )

        output = self.fine_acoustic_transformer.generate(
            torch.cat((coarse_tokens, fine_acoustic_tokens), dim=1)
        )

        return output

    def from_pretrained():

        semantic_encoder = W2VHuBERT_Quantizier()
        semantic_transformer = SemanticTransformer(
            500,  # Valore massimo del quantizzatore
            768,  # Numero arbitratio
        )
        dataloader = AudioDataLoader(os.getcwd() + "\\..\\data\\datasets", batch_size=1)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(semantic_transformer.parameters(), lr=0.001)
        intervals = 10
        save_path = Path(os.getcwd() + "\\..\\data\\")
        early_stop_counter = 10
        early_stopping_range = 10
        epochs = 1
        semantic_trainer = SemanticTrainer(
            semantic_encoder=semantic_encoder,
            semantic_transformer=semantic_transformer,
            train_dataloader=dataloader,
            val_dataloader=dataloader,
            test_dataloader=dataloader,
            loss=loss,
            optimizer=optimizer,
            intervals=intervals,
            save_path=save_path,
            early_stop_counter=early_stop_counter,
            early_stopping_range=early_stopping_range,
            epochs=epochs,
        )

        semantic_trainer.train()

    # def pipeline()
