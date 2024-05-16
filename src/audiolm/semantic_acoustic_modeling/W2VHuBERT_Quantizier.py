from transformers import AutoProcessor, HubertForCTC

# from sklearn.cluster import KMeans

import torch
import torchaudio
from torch import nn
import torchaudio.functional as F

from audiolm.data_preparation import AudioDataLoader


from pathlib import Path
import json
import os
import tqdm
import fairseq
from sklearn.cluster import KMeans
import joblib
import logging
from math import ceil

##Utils functions


def load_checkpoint():
    logging.getLogger("fairseq").setLevel(logging.WARNING)
    config_path = os.getcwd() + r"/src/audiolm/semantic_acoustic_modeling/config.json"
    assert Path(config_path).exists(), f"Config file not found in {config_path}"
    with open(config_path, "r") as f:
        config = json.load(f)
        checkpoint_path = config["checkpoint_path"]
        quantizier_path = config["quantizier_path"]
    if not os.path.exists("./cache"):
        os.makedirs("./cache")
    try:
        torch_check = torch.load(checkpoint_path)
        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            {checkpoint_path: torch_check}
        )
        model = models[0]
        print("Modello caricato")
    except Exception as e:
        raise Exception(f"An error occurred while trying to load the model: {e}")
    try:
        kmeans = joblib.load(quantizier_path)
        print("Kmeans caricato")
    except Exception as e:
        raise Exception(f"An error occurred while trying to load Kmeans: {e}")

    return model, kmeans


##Model class


class W2VHuBERT_Quantizier(nn.Module):
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
        self.model, self.kmeans = load_checkpoint()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dataloader = dataloader
        self.model.eval()
        self.sample_frequency = sample_frequency
        self.input_audio_hz = input_audio_hz
        self.layer = 6
        self.clusters = torch.from_numpy(self.kmeans.cluster_centers_)

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
        print(input_audio.shape)
        with torch.no_grad():
            embeddings = self.model(
                input_audio,
                mask=False,
                features_only=True,
                output_layer=self.layer,
            )["x"]
            print(embeddings.shape)
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
        for batch in tqdm.tqdm(
            self.dataloader,
            total=ceil(self.dataloader.__len__() / self.dataloader.batch_size),
        ):
            # print(batch.shape)
            batch = batch.squeeze(1)
            # print(batch.shape)
            batch = batch.to(self.device)
            out = self.forward(batch)
            semantic_tokens.append(out)

        return semantic_tokens


##Just for test


# data = "/Users/valerio/Desktop/ei/exterminationamericanbison_12_hornaday_64kb_0032.flac"
# audio, sr = torchaudio.load(data)

# print("Ecco l'audio: ", audio.shape)

# #audio = audio.unsqueeze(0) #Simulo la batch size
# #print("Ecco l'audio ora: ", audio.shape)

# hq = W2VHuBERT_Quantizier(sr, sr)

# hq.forward(audio)


if __name__ == "__main__":
    import os
    from audiolm.data_preparation import AudioDataset, AudioDataLoader
    from pathlib import Path

    dataloader = AudioDataLoader(
        os.getcwd() + "\\data\\datasets\\", 1, max_length_audio=3
    )
    audio = next(iter(dataloader))[0]
    hubert = W2VHuBERT_Quantizier(dataloader=dataloader)
    print(hubert.forward(audio).shape)
