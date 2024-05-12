"""Test suite for tokenizer model."""

import unittest
import os
from audiolm.coarse_acoustic_modelling.custom_encodec import CustomEncodecModel
from audiolm.data_preparation import AudioDataLoader


class TestEncodec(unittest.TestCase):
    dataloader = AudioDataLoader(
        data_path=os.getcwd() + "\\data\\datasets\\mini",
        batch_size=2,
        shuffle=False,
        max_length_audio=20,
        sample_frequency=24000,
    )
    encodec = CustomEncodecModel()
    # def test_encode(self):
