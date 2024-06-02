"""Unittests for trainer class."""

import os
import unittest
import unittest.test
import warnings
from pathlib import Path

import torch
from torch import nn

from audiolm.data_preparation import AudioDataLoader
from audiolm.w2v_hubert import W2VHuBert
from audiolm.absolute_transformer import (
    SemanticTransformer,
    CoarseAcousticTransformer,
    FineAcousticTransformer,
)
from audiolm.trainer import SemanticTrainer, CoarseAcousticTrainer, FineAcousticTrainer
from audiolm.encodec import Encodec

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.set_warn_always(False)

SAVE_PATH = Path(os.getcwd() + "\\data\\datasets")
MODEL_PATH = Path(os.getcwd() + "\\data")


class TestSemanticTransformerTrainer(unittest.TestCase):
    """Tests for Semantic Encoder trainer."""

    def test_train_end2end(self):
        """Test if the semantic trainer can be correctly trained"""
        print("===========================================")
        print("End to end Pipeline for semantic modelling.")
        print("===========================================")

        print("Created generator.")
        semantic_encoder = W2VHuBert()
        print("Created encoder.")

        semantic_transformer = SemanticTransformer()
        print("Created transformer.")
        dataloader = AudioDataLoader(os.getcwd() + "\\data\\datasets", batch_size=1)
        print("Fetched Dataloader and divided in train, test and validation.")
        loss = nn.CrossEntropyLoss()
        print("Instantiated Cross Entropy Loss")
        optimizer = torch.optim.Adam(semantic_transformer.parameters(), lr=0.001)
        print("Instantiated Adam")
        intervals = 10
        save_path = SAVE_PATH
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


class TestCoarseAcousticTransformerTrainer(unittest.TestCase):
    """Tests for Semantic Encoder trainer."""

    def test_train_end2end(self):
        """Test if the coarse acoustic trainer can be correctly trained"""
        print("===========================================")
        print("End to end Pipeline for coarse acoustic training.")
        print("===========================================")

        print("Created generator.")
        semantic_encoder = W2VHuBert()
        print("Created encoder.")
        semantic_transformer = SemanticTransformer()
        state_dict = torch.load(
            SAVE_PATH / "models" / f"{str(type(semantic_transformer).__name__)}.pth"
        )
        semantic_transformer.load_state_dict(state_dict)
        acoustic_encoder_decoder = Encodec()
        print("Created semantic transformer.")
        coarse_acoustic_transformer = CoarseAcousticTransformer()
        print("Created acoustic transformer.")
        dataloader = AudioDataLoader(os.getcwd() + "\\data\\datasets")
        print("Fetched Dataloader and divided in train, test and validation.")
        loss = nn.CrossEntropyLoss()
        print("Instantiated Cross Entropy Loss")
        optimizer = torch.optim.Adam(coarse_acoustic_transformer.parameters(), lr=0.001)
        print("Instantiated Adam")
        intervals = 10
        save_path = SAVE_PATH
        early_stop_counter = 10
        early_stopping_range = 10
        epochs = 1
        coarse_acoustic_trainer = CoarseAcousticTrainer(
            semantic_encoder=semantic_encoder,
            semantic_transformer=semantic_transformer,
            acoustic_encoder_decoder=acoustic_encoder_decoder,
            coarse_acoustic_transformer=coarse_acoustic_transformer,
            train_dataloader=dataloader,
            val_dataloader=dataloader,
            test_dataloader=None,
            loss=loss,
            optimizer=optimizer,
            intervals=intervals,
            save_path=save_path,
            early_stop_counter=early_stop_counter,
            early_stopping_range=early_stopping_range,
            epochs=epochs,
        )

        coarse_acoustic_trainer.train()


class TestFineAcousticTransformerTrainer(unittest.TestCase):
    """Tests for Semantic Encoder trainer."""

    def test_train_end2end(self):
        """Test if the coarse acoustic trainer can be correctly trained"""
        print("===========================================")
        print("End to end Pipeline for coarse acoustic training.")
        print("===========================================")

        print("Created generator.")
        semantic_encoder = W2VHuBert()
        print("Created encoder.")
        semantic_transformer = SemanticTransformer()
        state_dict = torch.load(
            SAVE_PATH / "models" / f"{str(type(semantic_transformer).__name__)}.pth"
        )
        semantic_transformer.load_state_dict(state_dict)
        acoustic_encoder_decoder = Encodec()
        print("Instantiated pretrained semantic transformer.")
        coarse_acoustic_transformer = CoarseAcousticTransformer()
        state_dict = torch.load(
            SAVE_PATH
            / "models"
            / f"{str(type(coarse_acoustic_transformer).__name__)}.pth"
        )
        coarse_acoustic_transformer.load_state_dict(state_dict)
        print("Instantiated pretrained coarse acoustic transformer.")
        fine_acoustic_transformer = FineAcousticTransformer()
        print("Instantiated pretrained fine acoustic transformer.")

        dataloader = AudioDataLoader(os.getcwd() + "\\data\\datasets")
        print("Fetched Dataloader and divided in train, test and validation.")
        loss = nn.CrossEntropyLoss()
        print("Instantiated Cross Entropy Loss")
        optimizer = torch.optim.Adam(fine_acoustic_transformer.parameters(), lr=0.001)

        intervals = 10
        save_path = Path(SAVE_PATH)
        early_stop_counter = 10
        early_stopping_range = 10
        epochs = 1
        fine_acoustic_trainer = FineAcousticTrainer(
            semantic_encoder=semantic_encoder,
            semantic_transformer=semantic_transformer,
            acoustic_encoder_decoder=acoustic_encoder_decoder,
            coarse_acoustic_transformer=coarse_acoustic_transformer,
            fine_acoustic_transformer=fine_acoustic_transformer,
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

        fine_acoustic_trainer.train()


if __name__ == "__main__":
    unittest.main()
