"""
Create a dummy LSTM, save/reload it
"""

import absl
import argparse
import keras
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
import tensorflow as tf
import keras.backend as K
from keras.models import load_model, Model
from keras.layers import BatchNormalization, LSTM, Dropout, Dense,\
        TimeDistributed, Masking, Activation, Input, Reshape, Embedding,\
        Bidirectional
from keras import regularizers
from keras.callbacks import ModelCheckpoint

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

absl.logging.set_verbosity(absl.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"

def main(cfg):
    # First save and restore basic LSTM model
    data = np.array([0.1,0.2,0.3]).reshape([1,3,1])
    inputs1 = Input(shape=data.shape[1:])
    lstm1 = LSTM(1, return_state=False, return_sequences=False)(inputs1)
    model1 = Model(inputs=inputs1, outputs=lstm1)
    out = model1.predict(data)
    print(f"model1 orignal output shape: {out.shape}")

    model_path = pj(HOME, "temp", "model_path")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cfg = vars( parser.parse_args() )
    main(cfg)

