"""
Run inference and analytics on mortality predictions
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys

import keras
import tensorboard
import tensorflow as tf

from datetime import datetime
from keras.layers import Bidirectional, Input
from keras.models import Model, model_from_yaml
from keras.utils import multi_gpu_model
from scipy import interp
from sklearn.manifold import Isomap, TSNE
from sklearn.metrics import roc_curve, auc,confusion_matrix, \
        average_precision_score, matthews_corrcoef
from sklearn.model_selection import KFold
from tensorflow.compat.v1.keras import backend as K

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from models import data_reader, evaluation, metrics
from data_extraction.data_extraction_mortality \
        import data_extraction_mortality
from data_extraction.utils import normalize_data_mort as normalize_data

from general.utils import plot_confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

# Turning these on made the model not run!!!
#tf.config.run_functions_eagerly(True)
#tf.compat.v1.disable_eager_execution()

def get_analysis_subsets(X_test, Y_test, N):
    alive = (Y_test==0).flatten()
    dead = (Y_test==1).flatten()
    X_alive = X_test[alive]
    X_dead = X_test[dead]
    inc_a = len(X_alive) // N
    X_alive = X_alive[::inc_a][:N] if inc_a>0 else X_alive[:N]
    inc_d = len(X_dead) // N
    X_dead = X_dead[::inc_d][:N] if inc_d>0 else X_dead[:N]
    X_anl = np.concatenate([X_alive, X_dead], axis=0)
    Y_anl = np.concatenate( [np.zeros(len(X_alive)), np.ones(len(X_dead))],
            axis=0 )
    return X_anl, Y_anl

def get_data(cfg):
    df_data = data_extraction_mortality(bm_config)
    all_idx = np.array(list(df_data['patientunitstayid'].unique()))
    skf = KFold(n_splits=bm_config.k_fold)

    folds = skf.split(all_idx)
    train_idx,test_idx = next(folds)
    train_idx = all_idx[train_idx]
    test_idx = all_idx[test_idx]

    train,test = normalize_data(bm_config, df_data, train_idx, test_idx)
    _,_,(X_test, Y_test),_ = data_reader.read_data(bm_config, train, test,
            val=False)
    return X_test, Y_test

def load_model(model_dir):
    yaml_file = pj( model_dir, "model_arch.yml" )
    model = model_from_yaml( open(yaml_file) )
    with open( pj(model_dir, "models", "checkpoint") ) as fp:
        line = next(fp).strip()
        start_idx = line.index("\"") + 1
        end_idx = len(line) - 1
        ckpt = line[ start_idx : end_idx ]
    load_status = model.load_weights( pj(model_dir, "models", ckpt) )
    load_status.expect_partial()
    return model


def main(cfg):
    model_dir = os.path.abspath( cfg["model_dir"] )
    X_test,Y_test = get_data(cfg)
    print(f"Data loaded. X_test shape: {X_test.shape}, Y_test shape: " \
            "{Y_test.shape}")

    model = load_model(model_dir)
    model.summary()
    print("Model loaded")

    probas_test = model.predict( [X_test[:,:,7:], X_test[:,:,:7]] )
    ix_pred_a = (probas_test < 0.5).flatten()
    ix_pred_d = (probas_test >= 0.5).flatten()
    ix_a = (Y_test==0).flatten()
    ix_d = (Y_test==1).flatten()
    ix_tn = ix_a & ix_pred_a
    ix_fp = ix_a & ix_pred_d
    ix_fn = ix_d & ix_pred_a
    ix_tp = ix_d & ix_pred_d
    X_anl,Y_anl = get_analysis_subsets(X_test, Y_test, cfg["num_for_analysis"])

    if cfg["write_out"]:
        pickle.dump(X_test, open(pj(bm_config.output_dir, "X_test.pkl"), "wb"))
        pickle.dump(Y_test, open(pj(bm_config.output_dir, "Y_test.pkl"), "wb"))
        # Note, data are *right-padded*, i.e. padded with zeros to the right
        # if there < 200 actual data samples
        # Y_test is {0,1}, 1 = death, about 12% mortality

    if cfg["cluster"]:
        bilstm_name = "bilstm_2"
        bilstm_layer = model.get_layer(bilstm_name)
        bilstm_layer.return_sequences = True
        bilstm_model = Model(inputs=model.input, outputs=bilstm_layer.output)
        bilstm_seqs = bilstm_model.predict( [X_test[:,:,7:], X_test[:,:,:7]] )
        print("Shape of BiLSTM output:", bilstm_seqs.shape)
        bilstm_seqs = np.concatenate( [bilstm_seqs[:,:,:64], 
            bilstm_seqs[:,::-1,64:]], axis=2 )

        reducer = cfg["reducer"]
        if reducer=="tsne":
            red_model = TSNE(n_components=2)
        elif reducer=="isomap":
            red_model = Isomap(n_components=2, n_neighbors=cfg["n_neighbors"])
        else:
            raise NotImplementedError(reducer)
        probas_out = bilstm_seqs[:,-1,:]
        print("Shape of final probas matrix:", probas_out.shape)
        print(f"Fitting {reducer} model...")
        proj_X = red_model.fit_transform(probas_out)
            # Should really be training tsne with training data but oh well
        print("...Done")
        
        plt.figure(figsize=(16,16))
        plt.scatter(proj_X[ix_tn,0], proj_X[ix_tn,1], s=12, c="r")
        plt.scatter(proj_X[ix_fn,0], proj_X[ix_fn,1], s=12, c="g")
        plt.scatter(proj_X[ix_fp,0], proj_X[ix_fp,1], s=12, c="y")
        plt.scatter(proj_X[ix_tp,0], proj_X[ix_tp,1], s=12, c="b")
        plt.savefig( pj(model_dir, f"{reducer}.png") )
        plt.close()

        inc = cfg["plot_every_nth"]
        slices_dir = pj(model_dir, f"{reducer}_slices")
        if not pe(slices_dir):
            os.makedirs(slices_dir)
        for j in range( bilstm_seqs.shape[1] ):
            slice_j = bilstm_seqs[::inc,j,:]
            proj_X_j = red_model.transform(slice_j)
            plt.figure(figsize=(16,16))
            plt.scatter( proj_X_j[ix_tn[::inc],0], proj_X_j[ix_tn[::inc],1],
                    s=12, c="r" )
            plt.scatter( proj_X_j[ix_fn[::inc],0], proj_X_j[ix_fn[::inc],1],
                    s=24, c="g" )
            plt.scatter( proj_X_j[ix_fp[::inc],0], proj_X_j[ix_fp[::inc],1],
                    s=12, c="y" )
            plt.scatter( proj_X_j[ix_tp[::inc],0], proj_X_j[ix_tp[::inc],1],
                    s=24, c="b" )
            plt.savefig( pj(slices_dir, f"{reducer}_{j:03d}.png") )
            plt.close()



    # Uses all subjects
    if cfg["confusion_matrix"]:
        print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
        print(f"Inferred probabilities, output shape {probas_test.shape}")

        fpr_mort, tpr_mort, thresholds = roc_curve(Y_test, probas_test)
        roc_auc_mort = auc(fpr_mort, tpr_mort)
        TN,FP,FN,TP = confusion_matrix(Y_test,probas_test.round()).ravel()
        PPV = TP/(TP+FP)
        NPV = TN/(TN+FN)

        cm = np.array( [[TN, FP], [FN, TP]] )
        save_path = pj( cfg["model_dir"], "confusion_matrix.png" )
        classes = ["False", "True"]
        plot_confusion_matrix(cm, save_path, classes,
                              normalize=False,
                              title='Confusion matrix')

        print("Inference:")
        print(f"PPV: {PPV:0.4f}, NPV: {NPV:0.4f}, roc_auc: " \
                "{roc_auc_mort:0.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str,
            default=pj(HOME, "Training/eICU_benchmark/mort_20200901-231037"))
#            default=pj(HOME, "Training/eICU_benchmark/mort_20200908-075827"))

    parser.add_argument("--task", default='mort', type=str, required=False)
    parser.add_argument("--num", default=True, type=str, required=False)
    parser.add_argument("--cat", default=True, type=str, required=False)
    parser.add_argument("--ann", default=False, type=str, required=False)
    parser.add_argument("--ohe", default=False, type=str, required=False)
    parser.add_argument("--mort_window", default=48, type=int, required=False)

    parser.add_argument("-N", "--num-for-analysis", type=int, default=100)
    parser.add_argument("-w", "--write-out", action="store_true",
            help="Write out X_test, Y_test")
    parser.add_argument("--cm", "--confusion-matrix", dest="confusion_matrix",
            action="store_true")
    parser.add_argument("--cluster", action="store_true",
            help="Cluster the probabilities, color-coded by TF/PN")
    parser.add_argument("-r", "--reducer", default="isomap",
            choices=["tsne", "isomap"])
    parser.add_argument("--n-neighbors", type=int, default=12)
    parser.add_argumnet("--nth", "--plot-every-nth", type=int, default=20)

    args = parser.parse_args()
    bm_config = Config(args)
    cfg = vars(args)
    main(cfg)

