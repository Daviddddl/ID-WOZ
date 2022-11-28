# install the requirements
# sudo python -m deeppavlov install intents_dstc2.json

# download pre-trained models
# sudo python -m deeppavlov download intents_dstc2.json

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from deeppavlov import train_model, configs

CONFIG_PATH = configs.classifiers.intents_dstc2

model = train_model(CONFIG_PATH, download=True)