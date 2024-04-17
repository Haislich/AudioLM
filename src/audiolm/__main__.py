"""Entry point for the app.

This can be invoked either by using

python -m audiolm

or via as a cli application (TODO) by using 

audiolm-cli
"""

from audiolm.app import AudioLMApp


if __name__ == "__main__":
    app = AudioLMApp()
    app.run()
