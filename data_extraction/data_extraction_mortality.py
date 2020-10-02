from __future__ import absolute_import
from __future__ import print_function


import argparse
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_extraction import utils
from config import Config

import pandas as pd
def data_extraction_mortality(args):
    time_window = args.mort_window
    all_df = utils.embedding(args.root_dir)
    all_mort = utils.filter_mortality_data(all_df)
    all_mort = all_mort[all_mort['itemoffset']<=time_window]
    return all_mort


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

def main():
    from keras.models import Model
    from sklearn.model_selection import KFold
    from data_extraction.utils import normalize_data_mort as normalize_data
    from models import data_reader
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    config = Config(args)
    df_data = data_extraction_mortality(config)
    print(f"Data shape: {df_data.shape}")
    print(f"Unique PIDs: {len(df_data.patientunitstayid.unique())}")
    
    all_idx = np.array(list(df_data['patientunitstayid'].unique()))
    skf = KFold(n_splits=config.k_fold)

    train_idx,test_idx = next( skf.split(all_idx) )
    train_idx = all_idx[train_idx]
    test_idx = all_idx[test_idx]

    train, test = normalize_data(config, df_data, train_idx, test_idx)
    train_gen, train_steps, (X_test, Y_test), max_time_step_test \
            = data_reader.read_data(config, train, test, val=False)
    batch = next(train_gen)
    print("Printing training data shape:")
    for b in batch:
        print("\t len", len(b))
        for b_i in b[:10]:
            print("\t\t", b_i.shape)
        if len(b)>10:
            print("\t\t...")
    print("Test data shape:")
    print(X_test.shape, Y_test.shape)


if __name__ == '__main__':
    main()
    
