import argparse
import numpy as np
import os
import pandas as pd
import sys

from tensorflow.keras.utils import Sequence
from keras.models import Model
from keras.layers import Input, Dense
        
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import models.metrics as metrics
from config import Config

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


# Adapted from
# https://towardsdatascience.com
# /keras-data-generators-and-how-to-use-them-b69129ed779c
class DataGenerator(Sequence):
    def __init__(self, pt_ids, labels, bm_config, to_fit=True, shuffle=True):
        """Initialization

        :param pt_ids: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self._batch_size = bm_config.batch_size
        self._data_dir = pj(bm_config.dpsom_supdir, "time_grid_pts")
        self._indexes = None
        self._input_dim = bm_config.dpsom_input_dim
        self._labels = labels
        self._pt_ids = pt_ids
        self._shuffle = shuffle
        self._time_dim = bm_config.dpsom_time_dim
        self._to_fit = to_fit

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch

        :return: number of batches per epoch
        """
        return len(self._pt_ids) // self._batch_size

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self._indexes[index * self._batch_size:(index + 1) \
                * self._batch_size]

        # Generate data
        X = self._generate_X(indexes)

        if self._to_fit:
            y = self._generate_y(indexes)
            return X, y
        else:
            return X

    def get_pt_ids(self):
        return self._pt_ids

    def on_epoch_end(self):
        """Updates indexes after each epoch

        """
        self._indexes = np.arange(len(self._pt_ids))
        if self._shuffle == True:
            np.random.shuffle(self._indexes)

    def _generate_X(self, indexes):
        """Generates data containing batch_size images

        :param pt_ids_temp: list of label ids to load
        :return: batch of images
        """
        pt_ids = [self._pt_ids[i] for i in indexes]
        # Initialization
        X = np.zeros((self._batch_size, self._time_dim, self._input_dim))

        # Generate data
        for i, pt_id in enumerate(pt_ids):
            df = pd.read_csv( pj(self._data_dir, str(pt_id)+".csv") )
            X_i = df.to_numpy()
            if X_i.shape[1]==100:
                X_i = X_i[ :, [1] + list(range(3,100)) ]
            elif X_i.shape[1]==99:
                X_i = X_i[ :, [0] + list(range(2,99)) ]
            istart_idx = max(0, len(X_i) - self._time_dim)
            tstart_idx = max(0, self._time_dim - len(X_i))
            X[ i, tstart_idx: ] = X_i[ istart_idx:, : ]

        return X

    def _generate_y(self, indexes):
        """Generates data containing batch_size masks

        :param indexes: list of indexes
        :return: batch if masks
        """
        y = np.array( [self._labels[i] for i in indexes] )
        return y


def batch_generator(pt_ids, mort, bm_config):
    train_gen = DataGenerator(pt_ids, mort, bm_config, to_fit=True,
            shuffle=True)
    index = 0
    while True:
        X,y = train_gen[index]
        index += 1
        if index==len(train_gen):
            train_gen.on_epoch_end()
            index = 0
        yield X,y


def dpsom_extraction_mortality(bm_config):
    outcome_df = pd.read_csv( pj(bm_config.dpsom_supdir, "static.csv") )
    mort = outcome_df.apachepatientresult_predictedhospitalmortality
    pt_ids = outcome_df[ mort>=0 ].patientunitstayid.to_numpy()
    mort = mort[ mort>=0 ]
    assert( len(pt_ids) == len(mort) )
    train_gen = batch_generator(pt_ids, mort.to_numpy(), bm_config)
    return train_gen

def _toy_model(bm_config):
    num_t = bm_config.dpsom_time_dim
    num_ch = bm_config.dpsom_input_dim
    inp = Input( shape=(num_t,num_ch) )
    out = Dense(1)(inp)
    optim = metrics.get_optimizer(lr=bm_config.lr)
    model = Model(inputs=inp, outputs=out)
    model.compile(loss="mean_squared_error", optimizer=optim,
                metrics=[metrics.f1,metrics.sensitivity, metrics.specificity,
                    'accuracy'])            
    return model

def main(bm_config, cfg):
    train_gen = dpsom_extraction_mortality(bm_config)
    print( dir(train_gen) )
#    print(f"Number of batches per epoch: {len(train_gen)}")
    X,y = next(train_gen)
    print(X.shape, y.shape)
    model = _toy_model(bm_config)
    model.fit(train_gen, steps_per_epoch=25, epochs=cfg["num_test_epochs"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-e", "--num-test-epochs", type=int, default=1)
    args = parser.parse_args()
    bm_config = Config(args)
    cfg = vars(args)
    main(bm_config, cfg)

