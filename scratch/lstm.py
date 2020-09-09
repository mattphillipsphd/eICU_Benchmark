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
    num_units = cfg["num_units"]

    lstm1 = Bidirectional(LSTM(units=num_units,
        kernel_regularizer=regularizers.l2(0.01),
        name="lstm1",
        return_sequences=True))(inputs1)
    lstm1 = BatchNormalization()(lstm1)
    lstm1 = Dropout(0.3)(lstm1)

    lstm2 = LSTM(num_units, return_state=False, return_sequences=False,
            name="lstm2")(lstm1)
    model1 = Model(inputs=inputs1, outputs=lstm2)
    out = model1.predict(data)
    print(f"model1 original output shape: {out.shape}")

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
    lstm_layer = model2.get_layer("lstm2")
    print( "No return state/sequences:", lstm_layer(data) )
    return_state = cfg["return_state"]
    lstm_layer.return_state = return_state
    state_str = "state/" if return_state else ""
    lstm_layer.return_sequences = True
    print( f"With return {state_str}sequences:", lstm_layer(data) )

    model3 = Model(inputs=model2.input, outputs=lstm_layer.output)
    print("Creating new model and predicting:", model3.predict(data))

def bilstm(cfg):
    data = np.array([0.1,0.2,0.3]).reshape([1,3,1])
    inputs = Input(shape=data.shape[1:])
    num_units = cfg["num_units"]

    lstm1 = Bidirectional(LSTM(units=num_units,
        kernel_regularizer=regularizers.l2(0.01),
        name="lstm1",
        return_sequences=True))(inputs)
    lstm1 = BatchNormalization()(lstm1)
    lstm1 = Dropout(0.3)(lstm1)

    lstm2 = LSTM(num_units, return_state=False, return_sequences=False)
    bilstm2 = Bidirectional(lstm2, name="lstm2")(lstm1)
    model1 = Model(inputs=inputs, outputs=bilstm2)
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

    bilstm_layer = model2.get_layer("lstm2")
    print( "No return state/sequences:" )
    last_h = bilstm_layer(data)
    print(f"\tlast_h: {last_h}")

    if cfg["wrapped_bilstm"]:
        wrapped_bilstm(bilstm_layer, data, model2, cfg)
    elif cfg["bilstm_fb"]:
        bilstm_fb(bilstm_layer, data, model2, cfg)

def bilstm_fb(bilstm_layer, data, full_model, cfg):
    lstmf_layer = bilstm_layer.forward_layer
    lstmb_layer = bilstm_layer.backward_layer
    lstmf_layer.return_sequences = True
    lstmb_layer.return_sequences = True
    return_state = cfg["return_state"]
    lstmf_layer.return_state = return_state
    lstmb_layer.return_state = return_state
    state_str = "state/" if return_state else ""
    print( f"With return {state_str}sequences, forward LSTM:" )
    if return_state:
        seq_h,last_fh,last_fc = lstmf_layer(data)
        print(f"\tseq_h: {seq_h}")
        print(f"\tlast_fh: {last_fh}")
        print(f"\tlast_fc: {last_fc}")
    else:
        seq_h = lstmf_layer(data)
        print(f"\tseq_h: {seq_h}")
    print( f"With return {state_str}sequences, backward LSTM:" )
    if return_state:
        seq_h,last_bh,last_bc = lstmb_layer(data)
        print(f"\tseq_h: {seq_h}")
        print(f"\tlast_bh: {last_bh}")
        print(f"\tlast_bc: {last_bc}")
    else:
        seq_h = lstmb_layer(data)
        print(f"\tseq_h: {seq_h}")
     
def tutorial(cfg):
    # From https://keras.io/guides/working_with_rnns/
    encoder_vocab = 1000
    decoder_vocab = 2000

    encoder_input = Input(shape=(None,))
    encoder_embedded = Embedding(input_dim=encoder_vocab, output_dim=64)(\
            encoder_input)

    # Return states in addition to output
    output, state_h, state_c = LSTM(64, return_state=True, name="encoder")(\
            encoder_embedded)
    encoder_state = [state_h, state_c]

    decoder_input = Input(shape=(None,))
    decoder_embedded = Embedding(input_dim=decoder_vocab, output_dim=64)(\
            decoder_input)

    # Pass the 2 states to a new LSTM layer, as initial state
    decoder_output = LSTM(64, name="decoder")(\
            decoder_embedded, initial_state=encoder_state)
    output = Dense(10)(decoder_output)

    model = Model([encoder_input, decoder_input], output)
    model.summary()

def wrapped_bilstm(bilstm_layer, data, full_model, cfg):
    return_state = cfg["return_state"]
    bilstm_layer.return_state = return_state
    bilstm_layer.backward_layer.return_state = return_state
    bilstm_layer.forward_layer.return_state = return_state
    bilstm_layer.return_sequences = True
    bilstm_layer.forward_layer.return_sequences = True
    bilstm_layer.backward_layer.return_sequences = True
    state_str = "state/" if return_state else ""
    # Note that passing data directly to the bilstm_layer only works because
    # it is equivalent to the initial_state of the layer
    print( f"With return {state_str}sequences:" )
    if return_state:
        seq_h,last_fh,last_fc,last_bh,last_bc = bilstm_layer(data)
        print(f"\tseq_h: {seq_h}")
        print(f"\tlast_fh: {last_fh}")
        print(f"\tlast_fc: {last_fc}")
        print(f"\tlast_bh: {last_bh}")
        print(f"\tlast_bc: {last_bc}")
    else:
        seq_h = bilstm_layer(data)
        print(f"\tseq_h: {seq_h}")

    model = Model(inputs=full_model.input, outputs=bilstm_layer.output)
    print("Now getting outputs from the model:")
    if return_state:
        seq_h,last_fh,last_fc,last_bh,last_bc = model.predict(data)
        print(f"\tseq_h: {seq_h}")
        print(f"\tlast_fh: {last_fh}")
        print(f"\tlast_fc: {last_fc}")
        print(f"\tlast_bh: {last_bh}")
        print(f"\tlast_bc: {last_bc}")
    else:
        seq_h = model.predict(data)
        print(f"\tseq_h: {seq_h}")

def main(cfg):
    if cfg["basic_lstm"]:
        basic_lstm(cfg)
    if cfg["tutorial"]:
        tutorial(cfg)
    if cfg["wrapped_bilstm"] or cfg["bilstm_fb"]:
        bilstm(cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--basic-lstm", action="store_true")
    parser.add_argument("-w", "--wrapped-bilstm", action="store_true")
    parser.add_argument("-l", "--bilstm-fb", action="store_true")
    parser.add_argument("-t", "--tutorial", action="store_true")

    parser.add_argument("--return-state", action="store_true")
    parser.add_argument("--num-units", type=int, default=1)
    cfg = vars( parser.parse_args() )
    main(cfg)

