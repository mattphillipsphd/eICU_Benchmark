"""
Run inference and analytics on mortality predictions
"""

import argparse
import numpy as np
import os
import sys

import keras
import tensorboard
import tensorflow as tf

from datetime import datetime
from keras.models import model_from_yaml
from keras.utils import multi_gpu_model
from scipy import interp
from sklearn.metrics import roc_curve, auc,confusion_matrix, \
        average_precision_score, matthews_corrcoef
from sklearn.model_selection import KFold
from tensorflow.compat.v1.keras import backend as K

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from models import data_reader
from models import evaluation
from data_extraction.data_extraction_mortality \
        import data_extraction_mortality
from data_extraction.utils import normalize_data_mort as normalize_data
from models.models import build_network as network

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


def get_data(cfg):
    df_data = data_extraction_mortality(bm_config)
    all_idx = np.array(list(df_data['patientunitstayid'].unique()))
    skf = KFold(n_splits=bm_config.k_fold)

    folds = skf.split(all_idx)
    train_idx,test_idx = next(folds)
    train_idx = all_idx[train_idx]
    test_idx = all_idx[test_idx]

    train,test = normalize_data(bm_config, df_data,train_idx, test_idx)
    _,_,(X_test, Y_test),_ = data_reader.read_data(bm_config, train, test,
            val=False)
    return X_test, Y_test

def load_model(cfg):
    yaml_file = pj( cfg["model_dir"], "model_arch.yml" )
    model = model_from_yaml( open(yaml_file) )
    with open( pj(cfg["model_dir"], "models", "checkpoint") ) as fp:
        line = next(fp).strip()
        start_idx = line.index("\"") + 1
        end_idx = len(line) - 1
        ckpt = line[ start_idx : end_idx ]
    model.load_weights( pj(cfg["model_dir"], "models", ckpt) )
    return model


def main(cfg):
    X_test,Y_test = get_data(cfg)
    print("Data loaded")

    model = load_model(cfg)
    print("Model loaded")

    probas_mort = model.predict([X_test[:,:,7:],X_test[:,:,:7]])
    print("Inferred probabilities")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str,
            default=pj(HOME, "Training/eICU_benchmark/mort_20200901-231037"))

    parser.add_argument("--task", default='mort', type=str, required=False)
    parser.add_argument("--num", default=True, type=str, required=False)
    parser.add_argument("--cat", default=True, type=str, required=False)
    parser.add_argument("--ann", default=False, type=str, required=False)
    parser.add_argument("--ohe", default=False, type=str, required=False)
    parser.add_argument("--mort_window", default=48, type=int, required=False)

    args = parser.parse_args()
    bm_config = Config(args)
    cfg = vars(args)
    main(cfg)

