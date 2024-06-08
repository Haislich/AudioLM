"""Unittests for trainer class."""

import os
import unittest
import unittest.test
from pathlib import Path

import torch
from torch import nn

from audiolm.absolute_transformer import CoarseAcousticTransformer, SemanticTransformer
from audiolm.data_preparation import AudioDataLoader
from audiolm.encodec import Encodec
from audiolm.w2v_hubert import W2VHuBert
from audiolm.trainer import CoarseAcousticTrainer, SemanticTrainer
from audiolm.utils import (
    get_latest_checkpoint_path,
    get_model_path,
    load_checkpoint,
    load_model,
)

DATA_PATH = Path(os.getcwd()) / Path("data") / Path("datasets")
SAVE_LOAD_PATH = Path(os.getcwd()) / Path("data")
INTERVALS = 10
EARLY_STOP_COUNTER = 0
EARLY_STOPPING_RANGE = 5
EPOCHS = 2

TRAIN_SEMANTIC = False
"""Start training for the semantic model"""
RESUME_SEMANTIC_TRAINING = True
"""Start training from the an epoch"""

TRAIN_COARSE = False
"""Start training for the semantic model"""
RESUME_COARSE_TRAINING = True
"""Start training from the an epoch"""
GENERATE_AUDIO_LEN = 3
"""Len in seconds of the audio generated"""


class TestTransformerTrainer(unittest.TestCase):
    """Tests for Semantic Encoder trainer."""

    semantic_encoder = W2VHuBert()
    acoustic_encoder_decoder = Encodec()
    train_dataloader = AudioDataLoader(DATA_PATH / "train", batch_size=4, max_elems=3)
    val_dataloader = AudioDataLoader(DATA_PATH / "val", batch_size=4, max_elems=3)
    test_dataloader = AudioDataLoader(DATA_PATH / "test", batch_size=4, max_elems=2)

    def test_train_semantic_end2end(self):
        """Test if the semantic trainer can be correctly trained"""
        print("===========================================")
        print("End to end Pipeline for semantic modelling.")
        print("===========================================")

        semantic_transformer = SemanticTransformer(num_heads=8, layers=4, feedforward_dim=2048)
        semantic_loss = nn.CrossEntropyLoss()
        semantic_optimizer = torch.optim.Adam(
            semantic_transformer.parameters(), lr=0.001
        )

        semantic_transformer_root = (
            Path(SAVE_LOAD_PATH)
            / Path("models")
            / str(type(semantic_transformer).__name__)
        )
        semantic_transformer_path = get_model_path(semantic_transformer_root)
        checkpoint_path = get_latest_checkpoint_path(semantic_transformer_root)
        if not TRAIN_SEMANTIC and semantic_transformer_path:
            load_model(semantic_transformer, semantic_transformer_path)
        else:
            if RESUME_SEMANTIC_TRAINING and checkpoint_path:
                print("Starting from the last epoch")
                semantic_transformer, _, semantic_optimizer, _ = load_checkpoint(
                    semantic_transformer, semantic_transformer_root
                )
            # elif not RESUME_SEMANTIC_TRAINING:
            #     # Adapt to choose a given epoch
            #     semantic_transformer, _, semantic_optimizer, _ = ...
            semantic_trainer = SemanticTrainer(
                semantic_encoder=self.semantic_encoder,
                semantic_transformer=semantic_transformer,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                test_dataloader=self.test_dataloader,
                loss=semantic_loss,
                optimizer=semantic_optimizer,
                intervals=INTERVALS,
                save_path=SAVE_LOAD_PATH,
                early_stop_counter=EARLY_STOP_COUNTER,
                early_stopping_range=EARLY_STOPPING_RANGE,
                epochs=EPOCHS,
            )
            semantic_trainer.train()
            torch.cuda.empty_cache()

    def _train_coarse_end2end(self):
        """Test if the semantic trainer can be correctly trained"""
        print("===========================================")
        print("End to end Pipeline for semantic modelling.")
        print("===========================================")

        semantic_transformer = SemanticTransformer(num_heads=8, layers=4, feedforward_dim=2048)
        semantic_transformer_root = (
            Path(SAVE_LOAD_PATH)
            / Path("models")
            / str(type(semantic_transformer).__name__)
        )
        semantic_transformer_path = get_model_path(semantic_transformer_root)

        load_model(semantic_transformer, semantic_transformer_path)

        coarse_acoustic_transformer = CoarseAcousticTransformer(num_heads=8, layers=4, feedforward_dim=2048)
        coarse_loss = nn.CrossEntropyLoss()
        coarse_optimizer = torch.optim.Adam(
            coarse_acoustic_transformer.parameters(), lr=0.001
        )

        coarse_transformer_root = (
            Path(SAVE_LOAD_PATH)
            / Path("models")
            / str(type(coarse_acoustic_transformer).__name__)
        )
        coarse_transformer_path = get_model_path(coarse_transformer_root)
        checkpoint_path = get_latest_checkpoint_path(coarse_transformer_root)

        if not TRAIN_COARSE and coarse_transformer_path:
            load_model(coarse_acoustic_transformer, coarse_transformer_path)
        else:
            if RESUME_COARSE_TRAINING and checkpoint_path:
                print("Starting from the last epoch")
                semantic_transformer, _, coarse_optimizer, _ = load_checkpoint(
                    semantic_transformer, coarse_transformer_root
                )
            # elif not RESUME_SEMANTIC_TRAINING:
            #     # Adapt to choose a given epoch
            #     semantic_transformer, _, semantic_optimizer, _ = ...
            coarse_acoustic_trainer = CoarseAcousticTrainer(
                semantic_encoder=self.semantic_encoder,
                semantic_transformer=semantic_transformer,
                acoustic_encoder_decoder=self.acoustic_encoder_decoder,
                coarse_acoustic_transformer=coarse_acoustic_transformer,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                test_dataloader=self.test_dataloader,
                loss=coarse_loss,
                optimizer=coarse_optimizer,
                intervals=INTERVALS,
                save_path=SAVE_LOAD_PATH,
                early_stop_counter=EARLY_STOP_COUNTER,
                early_stopping_range=EARLY_STOPPING_RANGE,
                generate_audio_len=GENERATE_AUDIO_LEN,
                epochs=EPOCHS,
            )
            coarse_acoustic_trainer.train()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
