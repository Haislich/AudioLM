"""This module contains the definition of the W2V_Hubert model"""

import logging
import os
from math import ceil
import warnings
import fairseq
import joblib
import requests
import torch
import torchaudio.functional as F
from torch import nn
from tqdm.auto import tqdm

from audiolm.constants import CACHE_PATH, DEVICE
from audiolm.data_preparation import AudioDataLoader

# region: Utils functions

logging.getLogger("fairseq").setLevel(logging.CRITICAL)


def _load_checkpoint():
    checkpoint_path = CACHE_PATH / "W2V_Hubert" / "model"
    if not checkpoint_path.exists():
        os.makedirs(checkpoint_path)
    checkpoint = checkpoint_path / "hubert_base_ls960.pt"
    if not checkpoint.exists():
        with requests.get(
            r"https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt",
            stream=True,
            timeout=30,
        ) as response:
            response.raise_for_status()
            # Display a progress bar to get the sense how the download is going
            with tqdm(
                desc="Downloading W2V_Hubert checkpoint.",
                total=int(response.headers.get("content-length", 0)),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                with open(checkpoint, mode="wb") as file:
                    for chunk in response.iter_content(chunk_size=8196):
                        progress_bar.update(len(chunk))
                        file.write(chunk)

    quantizer_path = CACHE_PATH / "W2V_Hubert" / "quantizer"
    if not quantizer_path.exists():
        os.makedirs(quantizer_path)

    quantizer = quantizer_path / "hubert_base_ls960_L9_km500.bin"
    if not quantizer.exists():
        with requests.get(
            r"https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin",
            stream=True,
            timeout=30,
        ) as response:
            response.raise_for_status()
            # Display a progress bar to get the sense how the download is going
            with tqdm(
                desc="Downloading W2V_Hubert quantizer.",
                total=int(response.headers.get("content-length", 0)),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                with open(quantizer, mode="wb") as file:
                    for chunk in response.iter_content(chunk_size=8196):
                        progress_bar.update(len(chunk))
                        file.write(chunk)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        torch_check = torch.load(checkpoint)
        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            {str(checkpoint): torch_check}
        )
        model = models[0]
        # print("Modello caricato")
        kmeans = joblib.load(quantizer)
        # print("Kmeans caricato")

    return model, kmeans


##Model class


class W2VHuBert(nn.Module):
    """
    A class representing a quantizer based on the W2VHuBERT model.
    This class takes an input audio signal, quantizes it using the W2VHuBERT model,
    and returns the quantized output.

    Args:
        sample_frequency (int): The sample frequency of the input audio signal. Default is 16000.
        dataloader (torch.utils.data.DataLoader): The dataloader used for fitting the quantizer. Default is None.

    Attributes:
        model (torch.nn.Module): The W2VHuBERT model.
        kmeans (sklearn.cluster.KMeans): The KMeans clustering model used for quantization.
        device (torch.device): The device (CPU or GPU) used for computation.
        dataloader (torch.utils.data.DataLoader): The dataloader used for fitting the quantizer.
        sample_frequency (int): The sample frequency of the input audio signal.
        input_audio_hz (int): The input audio frequency, useful if the Dataset class samples at a different frequency.
        layer (int): The layer of the W2VHuBERT model used for quantization.
        clusters (torch.Tensor): The cluster centers used for quantization.
    """

    def __init__(
        self,
        sample_frequency=16000,
        input_audio_hz=16000,
        dataloader: AudioDataLoader = None,
    ):
        super().__init__()
        self.model, self.kmeans = _load_checkpoint()
        self.dataloader = dataloader
        self.sample_frequency = sample_frequency
        self.input_audio_hz = input_audio_hz
        self.layer = 6
        self.clusters = torch.from_numpy(self.kmeans.cluster_centers_).to(DEVICE)

    def forward(self, input_audio):
        """
        Perform forward pass of the quantizer.

        Args:
            input_audio (torch.Tensor): Input audio signal.

        Returns:
            torch.Tensor: Quantized output.
        """
        if self.input_audio_hz != self.sample_frequency:
            input_audio = F.resample(
                input_audio, self.input_audio_hz, self.sample_frequency
            )
        if input_audio.dim() == 3:
            input_audio = input_audio.squeeze(1)
        # print(input_audio.shape)
        with torch.no_grad():
            embeddings = self.model(
                input_audio,
                mask=False,
                features_only=True,
                output_layer=self.layer,
            )["x"]
            # print(embeddings.shape)
            expand_cluster = self.clusters.unsqueeze(0).expand(
                embeddings.size(0), -1, -1
            )
            # print(expand_cluster.shape)
            assert embeddings.size(0) == expand_cluster.size(0) and embeddings.size(
                2
            ) == expand_cluster.size(2)
            distance = -torch.cdist(embeddings, expand_cluster, p=2)
            quantized = distance.argmax(-1)
            # print(quantized.shape)
        return quantized

    def build_TokenDataset(self):
        """

        Fit the quantizer to the input data and return the quantized tokens.

        Returns:
            list: List of quantized tokens.
        """
        semantic_tokens = []
        for batch in tqdm(
            self.dataloader,
            total=ceil(len(self.dataloader) / self.dataloader.batch_size),
        ):
            # print(batch.shape)
            batch = batch.squeeze(1)
            out = self.forward(batch)
            semantic_tokens.append(out)

        return semantic_tokens
