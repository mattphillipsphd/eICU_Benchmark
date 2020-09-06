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
from keras.models import load_model, model_from_json, Model
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

def basic_lstm(cfg):
    data = np.array([0.1,0.2,0.3]).reshape([1,3,1])
    inputs1 = Input(shape=data.shape[1:])
    lstm1 = LSTM(1, return_state=False, return_sequences=False)(inputs1)
    model1 = Model(inputs=inputs1, outputs=lstm1)
    out = model1.predict(data)
    print(f"model1 orignal output shape: {out.shape}")

    model_stub = pj(HOME, "temp", "model1")
    # serialize model to JSON
    model_json = model1.to_json()
    with open(model_stub+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model1.save_weights(model_stub+".h5")
    print(f"Wrote model to {model_stub}")
    
    # load json and create model
    json_file = open(model_stub+'.json', 'r')
    model2_json = json_file.read()
    json_file.close()
    model2 = model_from_json(model2_json)
    # load weights into new model
    model2.load_weights(model_stub+".h5")
    print("Loaded model from disk")

    model2.summary()
    lstm_layer = model2.get_layer("lstm")
    print( "No return state/sequences:", lstm_layer(data) )
    lstm_layer.return_state = True
    lstm_layer.return_sequences = True
    print( "With return state/sequences:", lstm_layer(data) )

def wrapped_bilstm(cfg):
    data = np.array([0.1,0.2,0.3]).reshape([1,3,1])
    inputs1 = Input(shape=data.shape[1:])
    lstm1 = LSTM(1, return_state=False, return_sequences=False)
    bilstm1 = Bidirectional(lstm1)(inputs1)
    model1 = Model(inputs=inputs1, outputs=bilstm1)
    out = model1.predict(data)
    print(f"model1 orignal output shape: {out.shape}")

    model_stub = pj(HOME, "temp", "bi_model1")
    # serialize model to JSON
    model_json = model1.to_json()
    with open(model_stub+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model1.save_weights(model_stub+".h5")
    print(f"Wrote model to {model_stub}")
    
    # load json and create model
    json_file = open(model_stub+'.json', 'r')
    model2_json = json_file.read()
    json_file.close()
    model2 = model_from_json(model2_json)
    # load weights into new model
    model2.load_weights(model_stub+".h5")
    print("Loaded model from disk")

    model2.summary()
    bilstm_layer = model2.get_layer("bidirectional")
    print( "No return state/sequences:" )
    last_h = bilstm_layer(data)
    print(f"\tlast_h: {last_h}")
    
    bilstm_layer.return_state = True
    bilstm_layer.return_sequences = True
    bilstm_layer.forward_layer.return_state = True
    bilstm_layer.forward_layer.return_sequences = True
    bilstm_layer.backward_layer.return_state = True
    bilstm_layer.backward_layer.return_sequences = True
    print( "With return state/sequences:" )
    seq_h,last_fh,last_fc,last_bh,last_bc = bilstm_layer(data)
    print(f"\tseq_h: {seq_h}")
    print(f"\tlast_fh: {last_fh}")
    print(f"\tlast_fc: {last_fc}")
    print(f"\tlast_bh: {last_bh}")
    print(f"\tlast_bc: {last_bc}")

def main(cfg):
    if cfg["basic_lstm"]:
        basic_lstm(cfg)
    if cfg["wrapped_bilstm"]:
        wrapped_bilstm(cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--basic-lstm", action="store_true")
    parser.add_argument("-w", "--wrapped-bilstm", action="store_true")
    cfg = vars( parser.parse_args() )
    main(cfg)

