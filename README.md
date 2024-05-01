<!-- <a target="_blank" href="https://colab.research.google.com/github/Haislich/AudioLM">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> -->

# AudioLM

[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Colab badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haislich/AudioLM)(WIP not ready yet)

Implementation of AudioLM paper by google

## Dependencies

## Setting up the development environment

AudioLM relies on relies on the [Poetry build system](https://python-poetry.org/).

To install `poetry`

```bash
pip install poetry
```

Once poetry is installed poetry will manage the needed dependencies and set up the working environment.

```bash
git clone https://github.com/Haislich/AudioLM   # Clone the repo.
cd AudioLM                                      # Change directory.
poetry shell                                    # This creates the venv.
poetry install                                  # This install the required dependencies.
```

### Ubuntu

Some libraries might not work out of the box, so this packages are required:

```bash
sudo apt-get install libportaaudio2
```

## References - Jose'

- [Github guidelines for repository files](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/creating-a-default-community-health-file)
- [Causal padding](https://medium.com/@stevechange/a-quick-journey-through-conv1d-functions-from-tensorflow-to-pytorch-passing-via-scipy-part-3-bda48e253953)
- [Explanation of Residual Vector Quantization](https://drscotthawley.github.io/blog/posts/2023-06-12-RVQ.html)
- [What are codebooks](https://machinelearning.wtf/terms/codebook/#:~:text=A%20codebook%20is%20a%20fixed,space%20of%20dimension%20Rn%20.)
- [More on codebooks](https://ai.stanford.edu/blog/codebook-features/)
- [Transposed Convolution](https://d2l.ai/chapter_computer-vision/transposed-conv.html)
- [More on transposed convolution](https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11)
- [Even more on transposed convolution and some examples of dilations](https://medium.com/@marsxiang/convolutions-transposed-and-deconvolution-6430c358a5b6)
- [Convolutions in Autoregressive Neural Networks](https://www.kilians.net/post/convolution-in-autoregressive-neural-networks/)
