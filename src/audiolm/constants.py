"""This file contains definitions of costant that will be used throught the execution."""

from pathlib import Path

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""What device to use."""
DEBUG = True
"""Debugging flag."""
CACHE_PATH = Path.home() / ".cache/AudioLM"
"""Location of the cache, is assumed to be under `~/.cache`"""
