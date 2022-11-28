import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from deeppavlov import train_model, configs

CONFIG_PATH = configs.classifiers.intents_multiwoz_bahasa

model = train_model(CONFIG_PATH, download=True)