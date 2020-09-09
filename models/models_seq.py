"""
Just like models.py but builds a network that returns sequences
"""

import argparse
import json
import numpy as np
import os
import random
import sys

import keras
import keras.backend as K
import tensorflow as tf

from keras.models import load_model, Model
from keras.layers import BatchNormalization, LSTM, Dropout, Dense,\
        TimeDistributed, Masking, Activation, Input, Reshape, Embedding,\
        Bidirectional
from keras import regularizers
from keras.callbacks import ModelCheckpoint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import metrics
from config import Config

seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


# common network
def build_network_seq(config, input_size, output_dim=1, activation='sigmoid'):
    input1 = Input(shape=(input_size, 13))
    input2 = Input(shape=(input_size, 7))
    x2 = Embedding(config.n_cat_class, config.embedding_dim)(input2)
    x2 = Reshape((int(x2.shape[1]),int(x2.shape[2]*x2.shape[3])))(x2)
    inp = keras.layers.Concatenate(axis=-1)([input1, x2])

    mask = Masking(mask_value=0., name="maski")(inp)

    lstm = mask
    for i in range(config.rnn_layers-1):
        lstm = Bidirectional(LSTM(units=config.rnn_units[i],
            kernel_regularizer=regularizers.l2(0.01),
            kernel_initializer='glorot_normal', name="lstm_{}".format(i+1),
            return_sequences=True), name=f"bilstm_{i+1}")(lstm)
        lstm = BatchNormalization()(lstm)
        lstm = Dropout(config.dropout)(lstm)

    if config.task in ['rlos', 'dec']:
        lstm = Bidirectional(LSTM(units=config.rnn_units[-1],
            kernel_regularizer=regularizers.l2(0.01),
            kernel_initializer='glorot_normal', name="lstm_{}".format(
                config.rnn_layers), return_sequences=True))(lstm)
    elif config.task in ['mort']:
        N = config.rnn_layers
        lstm = Bidirectional(LSTM(units=config.rnn_units[-1],
            kernel_regularizer=regularizers.l2(0.01),
            kernel_initializer='glorot_normal', name="lstm_{}".format(N),
            return_sequences=True), name=f"bilstm_{N}")(lstm)
        lstm = tf.concat( [lstm[:,-1,:64], lstm[:,0,64:]], axis=1 )
        # With return_sequences=True, lstm.shape is (None, 200, 128) then
        # (None, 128)
        # With return_sequences=False, lstm.shape is (None,128) from the 
        # outset
    elif config.task in ['phen']:
        lstm = Bidirectional(LSTM(units=config.rnn_units[-1],
            kernel_regularizer=regularizers.l2(0.01),
            kernel_initializer='glorot_normal', name="lstm_{}".format(\
                    config.rnn_layers), return_sequences=False))(lstm)
    else:
        print('Invalid task type.')
        exit()

    lstm = BatchNormalization()(lstm)
    lstm = Dropout(config.dropout)(lstm)

    if config.task in ['rlos', 'dec']:
        out = TimeDistributed(Dense(output_dim, activation=activation))(lstm)
    
    elif config.task in ['mort', 'phen']:
        out = Dense(output_dim, activation=activation)(lstm)
    
    else:
        print('Invalid task type.')
        exit() 

    if config.num and config.cat:
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    else:
        model = keras.models.Model(inputs=input1, outputs=out)

    optim = metrics.get_optimizer(lr=config.lr)

    if config.task == 'mort':
        model.compile(loss="binary_crossentropy", optimizer=optim,
                metrics=[metrics.f1,metrics.sensitivity, metrics.specificity,
                    'accuracy'])
        # model.summary()
    elif config.task == 'rlos':
        model.compile(loss='mean_squared_error', optimizer=optim,
                metrics=['mse'])
    
    elif config.task in ['phen', 'dec']:
        model.compile(loss="binary_crossentropy" ,optimizer=optim,
                metrics=[metrics.f1,'accuracy'])
    
    else:
        print('Invalid task name')

    # print(model.summary())
    return model


def main(cfg):
    input_size = cfg["input_size"]
    full_model = build_network_seq(bm_config, input_size, output_dim=1, 
            activation='sigmoid')
    full_model.summary()
    X = np.random.rand(3,input_size,20)
    if cfg["test_masking"]:
        X[:,10:,:] = 0.0
    X = [X[:,:,7:], X[:,:,:7]]
    out = full_model.predict(X)
    print("Full model")
    print(len(out))
    for o in out:
        if input_size<20:
            print(o)
        print("\t", o.shape)

    bilstm_layer = full_model.get_layer("bidirectional_1")
    bilstm_model = Model(inputs=full_model.input, outputs=bilstm_layer.output)
    out_seq = bilstm_model.predict(X)
    print("BiLSTM model")
    print(len(out_seq))
    for o in out_seq:
        if input_size<20:
            print(o)
        print("\t", o.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='mort', type=str, required=False)
    parser.add_argument("--num", default=True, type=str, required=False)
    parser.add_argument("--cat", default=True, type=str, required=False)
    parser.add_argument("--ann", default=False, type=str, required=False)
    parser.add_argument("--ohe", default=False, type=str, required=False)
    parser.add_argument("--mort_window", default=48, type=int, required=False)

    parser.add_argument("-i", "--input-size", type=int, default=200)
    parser.add_argument("-m", "--test-masking", action="store_true")

    args = parser.parse_args()
    bm_config = Config(args)
    cfg = vars(args)
    main(cfg)

