# AudioLM

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

## References

- <https://colab.research.google.com/github/fastforwardlabs/whisper-openai/blob/master/WhisperDemo.ipynb#scrollTo=wIRFnTn3Fzua>
