import json
import os
from pathlib import Path

import torch
import torchaudio.functional as F
import tqdm
from sklearn.cluster import KMeans
from torch import nn
from transformers import AutoProcessor, Wav2Vec2BertModel

##Utils functions


def load_model():
    config_path = "./config.json"
    assert (Path(config_path).exists(), f"Config file not found in {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
        model_name = config["model_name"]
        n_clusters = config["n_clusters"]
        n_init = config["n_init"]
        k_means_verbose = config["kmeans_verbose"]

    if not os.path.exists("./cache"):
        os.makedirs("./cache")

    try:
        model = Wav2Vec2BertModel.from_pretrained(model_name, cache_dir="./cache")
        processor = AutoProcessor.from_pretrained(model_name, cache_dir="./cache")
    except Exception as e:
        raise Exception(f"An error occurred while trying to load the model: {e}")

    return model, processor, n_clusters, n_init, k_means_verbose


##Model class


class W2VBertKmeans(nn.Module):

    def __init__(self, sample_frequency=16000, dataloader=None, device=None):
        super().__init__()
        (
            self.model,
            self.processor,
            self.n_clusters,
            self.n_init,
            self.kmeans_verbose,
        ) = load_model()
        self.model.to(device)
        self.model.eval()
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            verbose=self.kmeans_verbose,
            n_init=self.n_init,
        )
        self.sample_frequency = sample_frequency
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer = (6,)
        self

    def forward(self, input_audio, input_audio_hz):
        input_audio = F.resample(input_audio, input_audio_hz, self.sample_frequency)
        with torch.no_grad():
            input_features = self.processor(
                input_audio, sampling_rate=self.sample_frequency, return_tensors="pt"
            )["input_features"]
            input_features = input_features.to(self.device)
            embedding = self.model(
                input_features, output_hidden_states=True, return_dict=True
            )
            seventh_layer_output = embedding.hidden_states[6].detach().cpu().numpy()

        return seventh_layer_output

    def fit(self, dataloader):
        embeddings = []
        for batch in tqdm.tqdm(dataloader):
            batch = batch.to(self.device)
            out = self.forward(batch, 24000)
            embeddings.append(out)
