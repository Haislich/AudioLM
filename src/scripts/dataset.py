"""This module is intended to host every function responsible for the datasets."""

import os
import pathlib
import tarfile
import requests
from tqdm.auto import tqdm


def get_dataset(
    url: str = r"https://www.openslr.org/resources/12/train-clean-100.tar.gz",
):
    """Fetches the dataset found at `url`.
    Downloads the compressed dataset and then uncompress it.
    This function assumes that the datasets are compressed as tar.gz/tar

    Args:
        url: The url where the dataset is hosted.
    """
    path = pathlib.Path("./datasets")
    compressed_filename = url.split("/")[-1]
    filename = compressed_filename.split(".")[0]
    if not (path / filename).exists():
        if not path.exists():
            os.mkdir(path)
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            # Display a progress bar to get the sense how the download is going
            with tqdm(
                desc=f"Downloading dataset: {compressed_filename}",
                total=int(response.headers.get("content-length", 0)),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                with open(path / compressed_filename, mode="wb") as file:
                    for chunk in response.iter_content(chunk_size=8196):
                        progress_bar.update(len(chunk))
                        file.write(chunk)
        with tarfile.open(path / compressed_filename) as compressed_file:
            for member in tqdm(
                desc=f"Extracting dataset {compressed_filename}",
                iterable=compressed_file.getmembers(),
                total=len(compressed_file.getmembers()),
            ):
                compressed_file.extract(member=member, path=path / filename)
        os.remove(path / compressed_filename)
    print(f"Dataset is ready at {path / filename}")


def get_checkpoint(
    url: str = r"https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt",
):
    """Fetches the dataset found at `url`.
    Downloads the compressed dataset and then uncompress it.
    This function assumes that the datasets are compressed as tar.gz/tar

    Args:
        url: The url where the dataset is hosted.
    """
    path = pathlib.Path("./datasets")
    compressed_filename = url.split("/")[-1]
    filename = compressed_filename.split(".")[0]
    if not (path / filename).exists():
        if not path.exists():
            os.mkdir(path)
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            # Display a progress bar to get the sense how the download is going
            with tqdm(
                desc=f"Downloading dataset: {compressed_filename}",
                total=int(response.headers.get("content-length", 0)),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                with open(path / compressed_filename, mode="wb") as file:
                    for chunk in response.iter_content(chunk_size=8196):
                        progress_bar.update(len(chunk))
                        file.write(chunk)
        with tarfile.open(path / compressed_filename) as compressed_file:
            for member in tqdm(
                desc=f"Extracting dataset {compressed_filename}",
                iterable=compressed_file.getmembers(),
                total=len(compressed_file.getmembers()),
            ):
                compressed_file.extract(member=member, path=path / filename)
        os.remove(path / compressed_filename)
    print(f"Dataset is ready at {path / filename}")


get_dataset("https://dl.fbaipublicfiles.com/librilight/data/small.tar")
