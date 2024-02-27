# AudioLM

Implementation of AudioLM paper by google

## Setting up the development environment

This package relies on the [Poetry build system](https://python-poetry.org/).

```bash
pip install poetry                              # Install poetry.
git clone https://github.com/Haislich/AudioLM   # Clone the repo.
cd AudioLM                                      # Change directory.
poetry shell                                    # This creates the venv.
poetry install                                  # This install the required dependencies.
```

On Ubuntu

```bash
sudo apt-get install libportaaudio2
```

## References

- <https://colab.research.google.com/github/fastforwardlabs/whisper-openai/blob/master/WhisperDemo.ipynb#scrollTo=wIRFnTn3Fzua>
