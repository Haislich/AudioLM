"""Unittests for trainer class."""

import os
import unittest
import unittest.test
import warnings
from pathlib import Path

import torch
from torch import nn

from audiolm.data_preparation import AudioDataLoader
from audiolm.semantic_acoustic_modeling.W2VHuBERT_Quantizier import W2VHuBERT_Quantizier
from audiolm.transformer.AbsoluteTransformer import TransformerDecoderOnly
from audiolm.transformer.trainer import SemanticTrainer, AcousticTrainer
from audiolm.custom_encodec import CustomEncodecModel

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.set_warn_always(False)


class TestSemanticEncoder(unittest.TestCase):
    def test_train_end2end(self):
        """Test if the pipeline generate a single emebeding"""
        print("===========================================")
        print("End to end Pipeline for semantic modelling.")
        print("===========================================")

        print("Created generator.")
        semantic_encoder = W2VHuBERT_Quantizier()
        print("Created encoder.")
        semantic_transformer = TransformerDecoderOnly(
            500,  # Valore massimo del quantizzatore
            768,  # Numero arbitratio
        )
        print("Created transformer.")
        dataloader = AudioDataLoader(os.getcwd() + "\\data\\datasets")
        print("Fetched Dataloader and divided in train, test and validation.")
        loss = nn.CrossEntropyLoss()
        print("Created cross entropy loss")
        optimizer = torch.optim.Adam(semantic_transformer.parameters(), lr=0.001)
        print("Adam")
        intervals = 10
        save_path = Path(os.getcwd() + "\\data\\models")
        early_stop_counter = 10
        early_stopping_range = 10
        epochs = 10
        semantic_trainer = SemanticTrainer(
            semantic_encoder=semantic_encoder,
            semantic_transformer=semantic_transformer,
            train_dataloader=dataloader,
            val_dataloader=None,
            test_dataloader=None,
            loss=loss,
            optimizer=optimizer,
            intervals=intervals,
            save_path=save_path,
            early_stop_counter=early_stop_counter,
            early_stopping_range=early_stopping_range,
            epochs=epochs,
        )

        semantic_trainer.train()


class TestAcousticEncoder(unittest.TestCase):
    def test_train_end2end(self):
        """Test if the pipeline generate a single emebeding"""
        print("===========================================")
        print("End to end Pipeline for coarse acoustic training.")
        print("===========================================")

        print("Created generator.")
        semantic_encoder = W2VHuBERT_Quantizier()
        print("Created encoder.")
        semantic_transformer = TransformerDecoderOnly(
            500,  # Valore massimo del quantizzatore
            768,  # Numero arbitratio
        )
        acoustic_enc_dec = CustomEncodecModel()
        print("Created semantic transformer.")
        acoustic_transformer = TransformerDecoderOnly(
            1024,  # Valore massimo del quantizzatore
            1024,  # Numero arbitratio
        )
        print("Created acoustic transformer.")
        dataloader = AudioDataLoader(os.getcwd() + "\\data\\datasets")
        print("Fetched Dataloader and divided in train, test and validation.")
        loss = nn.CrossEntropyLoss()
        print("Created cross entropy loss")
        optimizer = torch.optim.Adam(semantic_transformer.parameters(), lr=0.001)
        print("Adam")
        intervals = 10
        save_path = Path(os.getcwd() + "\\data\\models")
        early_stop_counter = 10
        early_stopping_range = 10
        epochs = 2
        semantic_trainer = AcousticTrainer(
            semantic_encoder=semantic_encoder,
            semantic_transformer=semantic_transformer,
            acoustic_enc_dec=acoustic_enc_dec,
            acoustic_transformer=acoustic_transformer,
            train_dataloader=dataloader,
            val_dataloader=None,
            test_dataloader=None,
            loss=loss,
            optimizer=optimizer,
            intervals=intervals,
            save_path=save_path,
            early_stop_counter=early_stop_counter,
            early_stopping_range=early_stopping_range,
            epochs=epochs,
        )

        semantic_trainer.train()


if __name__ == "__main__":
    unittest.main()
