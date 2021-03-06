import argparse
import numpy as np
import os
import sys

import keras
import tensorboard
import tensorflow as tf

from datetime import datetime
from keras.utils import multi_gpu_model
from scipy import interp
from sklearn.metrics import roc_curve, auc,confusion_matrix, \
        average_precision_score, matthews_corrcoef
from sklearn.model_selection import KFold
from tensorflow.compat.v1.keras import backend as K

from config import Config
from models import data_reader
from models import evaluation
from models.models import build_network as network
from models.models_seq import build_network_seq as network_seq

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")


g_session_str = None
def set_session_str(task):
    global g_session_str
    g_session_str = task + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

#Decompensation
def train_dec(config):
    from data_extraction.utils import normalize_data_dec as normalize_data
    from data_extraction.data_extraction_decompensation \
            import data_extraction_decompensation as extract_data
    df_data = extract_data(config)

    cvscores_dec = []
    tprs_dec = []
    aucs_dec = []
    mean_fpr_dec = np.linspace(0, 1, 100)
    i_dec = 0
    ppvs_dec = []
    npvs_dec = []
    aucprs_dec = []
    mccs_dec = []
    specat90_dec = []

    log_dir = pj( HOME, "tb-logs", g_session_str )
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
    models_dir = pj(config.output_dir, g_session_str, "models")
    os.makedirs(models_dir)

    all_idx = np.array(list(df_data['patientunitstayid'].unique()))

    skf = KFold(n_splits=config.k_fold)
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(all_idx)):
        print('Running Fold {}...'.format(fold_id+1))
        train_idx = all_idx[train_idx]
        test_idx = all_idx[test_idx]

        train, test = normalize_data(config, df_data,train_idx, test_idx)
        train_gen, train_steps, (X_test, Y_test), max_time_step_test \
                = data_reader.read_data(config, train, test, val=False)
     
        model = network(config, 200, output_dim=1, activation='sigmoid')

        prefix = f"fold_{fold_id}"
        checkpoint_path = pj(models_dir, prefix + "_cp-{epoch:04d}.ckpt")
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=1, 
            save_weights_only=True,
            save_freq=config.save_freq)

        history = model.fit(train_gen, steps_per_epoch=25,
                            epochs=config.epochs, verbose=1, shuffle=True,
                            callbacks=[tensorboard_callback, cp_callback])
        if config.num and config.cat:
            if config.ohe:
                x_cat = X_test[:, :, :7].astype(int)
                x_nc = X_test[:,:,7:]
                print("Please wait, One-hot encoding ...")
                one_hot = np.zeros((x_cat.shape[0], x_cat.shape[1], 429),
                        dtype=np.int)
                x_cat = (np.eye(config.n_cat_class)[x_cat].sum(2) > 0)\
                        .astype(int)
                probas_dec = model.predict([x_nc, x_cat])
                #todo Replace np.eye with faster function
            else:
                probas_dec = model.predict([X_test[:,:,7:],X_test[:,:,:7]])
        elif config.cat:
            if config.ohe:
                x_cat = X_test[:, :, :].astype(int)
                print("Please wait, One-hot encoding ...")
                one_hot = np.zeros((x_cat.shape[0], x_cat.shape[1], 429),
                        dtype=np.int)
                x_cat = (np.eye(config.n_cat_class)[x_cat].sum(2) > 0)\
                        .astype(int)
                probas_dec = model.predict([x_cat])
                #todo Replace np.eye with faster function
            else:
                probas_dec = model.predict([X_test])
        else:
            probas_dec = model.predict([X_test])

        Y_test, probas_dec = evaluation.decompensation_metrics(Y_test,
                probas_dec,max_time_step_test)
        
        fpr_dec, tpr_dec, thresholds = roc_curve(Y_test, probas_dec)
        specat90_dec.append(1-fpr_dec[tpr_dec>=0.90][0])
        tprs_dec.append(interp(mean_fpr_dec, fpr_dec, tpr_dec))
        tprs_dec[-1][0] = 0.0
        roc_auc_dec = auc(fpr_dec, tpr_dec)
        aucs_dec.append(roc_auc_dec)
        
        TN,FP,FN,TP = confusion_matrix(Y_test, probas_dec.round()).ravel()
        PPV = TP/(TP+FP)
        NPV = TN/(TN+FN)
        ppvs_dec.append(PPV)
        npvs_dec.append(NPV)
        
        average_precision_dec = average_precision_score(Y_test, probas_dec)
        aucprs_dec.append(average_precision_dec)
        
        mccs_dec.append(matthews_corrcoef(Y_test, probas_dec.round()))
    
    mean_tpr_dec = np.mean(tprs_dec, axis=0)
    mean_tpr_dec[-1] = 1.0
    mean_auc_dec = auc(mean_fpr_dec, mean_tpr_dec)
    std_auc_dec = np.std(aucs_dec)
    
    print("====================Decompensation================")
    print("Mean AUC{0:0.3f} +- STD{1:0.3f}".format(mean_auc_dec,std_auc_dec))
    print("PPV: {0:0.3f}".format(np.mean(ppvs_dec)))
    print("NPV: {0:0.3f}".format(np.mean(npvs_dec)))
    print("AUCPR:{0:0.3f}".format(np.mean(aucprs_dec)))
    print("MCC: {0:0.3f}".format(np.mean(mccs_dec)))
    print("Spec@90: {0:0.3f}".format(np.mean(specat90_dec)))

    return {'Mean AUC': mean_auc_dec,
            'STD':std_auc_dec,
            'PPV':np.mean(ppvs_dec),
            'NPV':np.mean(npvs_dec),
            'AUCPR':np.mean(aucprs_dec),
            'MCC':np.mean(mccs_dec),
            'Spec@90':np.mean(specat90_dec)}

def write_summary(session_dir, model):
    mstr = model.to_yaml()
    with open( pj(session_dir, "model_arch.yml"), "w" ) as fp:
        fp.write(mstr)

# Mortality using DPSOM paper preprocessing
def train_dpsom_mort(config):
    from data_extraction.dpsom_mort import dpsom_extraction_mortality
    train_gen,train_steps,test_gen,test_steps = dpsom_extraction_mortality(\
            config)
    time_dim = config.dpsom_time_dim
    model = network_seq(config, time_dim, output_dim=1, activation='sigmoid')
    write_summary( pj(config.output_dir, g_session_str), model)

    models_dir = pj(config.output_dir, g_session_str, "models")
    os.makedirs(models_dir)
    fold_id = 0

    prefix = f"fold_{fold_id}"
    checkpoint_path = pj(models_dir, prefix + "_cp-{epoch:04d}.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq=config.save_freq)

    history = model.fit(train_gen, steps_per_epoch=25,
            epochs=config.epochs,
            callbacks=[cp_callback,])
    model.predict(test_gen)

#Mortality
def train_mort(config):
    cvscores_mort = []
    tprs_mort = []
    aucs_mort = []
    mean_fpr_mort = np.linspace(0, 1, 100)
    i_mort = 0
    ppvs_mort = []
    npvs_mort = []
    aucprs_mort = []
    mccs_mort = []
    specat90_mort = []

    from data_extraction.data_extraction_mortality \
            import data_extraction_mortality
    from data_extraction.utils import normalize_data_mort as normalize_data

    df_data = data_extraction_mortality(config)
    all_idx = np.array(list(df_data['patientunitstayid'].unique()))
    skf = KFold(n_splits=config.k_fold)

    log_dir = pj( HOME, "tb-logs", g_session_str )
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
    models_dir = pj(config.output_dir, g_session_str, "models")
    os.makedirs(models_dir)

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(all_idx)):
        print('Running Fold {}...'.format(fold_id+1))
        train_idx = all_idx[train_idx]  
        test_idx = all_idx[test_idx]

        train, test = normalize_data(config, df_data, train_idx, test_idx)
        train_gen, train_steps, (X_test, Y_test), max_time_step_test \
                = data_reader.read_data(config, train, test, val=False)

#        model = network(config, 200, output_dim=1, activation='sigmoid')
        model = network_seq(config, 200, output_dim=1, activation='sigmoid')
        write_summary( pj(config.output_dir, g_session_str), model)

        prefix = f"fold_{fold_id}"
        checkpoint_path = pj(models_dir, prefix + "_cp-{epoch:04d}.ckpt")
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=1, 
            save_weights_only=True,
            save_freq=config.save_freq)

        history = model.fit(train_gen, steps_per_epoch=25,
                epochs=config.epochs, verbose=1, shuffle=True,
                callbacks=[tensorboard_callback, cp_callback])
        
        if config.num and config.cat:
            if config.ohe:
                x_cat = X_test[:, :, :7].astype(int)
                x_nc = X_test[:,:,7:]
                print("Please wait, One-hot encoding ...")
                one_hot = np.zeros((x_cat.shape[0], x_cat.shape[1], 429),
                        dtype=np.int)
                x_cat = (np.eye(config.n_cat_class)[x_cat].sum(2) > 0)\
                        .astype(int)
                probas_mort = model.predict([x_nc, x_cat])
                #todo Replace np.eye with faster function
            else:
                probas_mort = model.predict([X_test[:,:,7:],X_test[:,:,:7]])
        elif config.cat:
            if config.ohe:
                x_cat = X_test[:, :, :].astype(int)
                print("Please wait, One-hot encoding ...")
                one_hot = np.zeros((x_cat.shape[0], x_cat.shape[1], 429),
                        dtype=np.int)
                x_cat = (np.eye(config.n_cat_class)[x_cat].sum(2) > 0)\
                        .astype(int)
                probas_mort = model.predict([x_cat])
                #todo Replace np.eye with faster function
            else:
                probas_mort = model.predict([X_test])
        else:
            probas_mort = model.predict([X_test])

        
        fpr_mort, tpr_mort, thresholds = roc_curve(Y_test, probas_mort)
        tprs_mort.append(interp(mean_fpr_mort, fpr_mort, tpr_mort))
        tprs_mort[-1][0] = 0.0
        roc_auc_mort = auc(fpr_mort, tpr_mort)
        aucs_mort.append(roc_auc_mort)
        TN,FP,FN,TP = confusion_matrix(Y_test,probas_mort.round()).ravel()
        PPV = TP/(TP+FP)
        NPV = TN/(TN+FN)

        ppvs_mort.append(PPV)
        npvs_mort.append(NPV)
        average_precision_mort = average_precision_score(Y_test,probas_mort)
        aucprs_mort.append(average_precision_mort)
        mccs_mort.append(matthews_corrcoef(Y_test, probas_mort.round()))
        specat90_mort.append(1-fpr_mort[tpr_mort>=0.90][0])

    mean_tpr_mort = np.mean(tprs_mort, axis=0)
    mean_tpr_mort[-1] = 1.0
    mean_auc_mort = auc(mean_fpr_mort, mean_tpr_mort)
    std_auc_mort = np.std(aucs_mort)

    print("===========================Mortality=============================")
    print("Mean AUC {0:0.3f} +- STD{1:0.3f}".format(mean_auc_mort,
        std_auc_mort))
    print("PPV: {0:0.3f}".format(np.mean(ppvs_mort)))
    print("NPV: {0:0.3f}".format(np.mean(npvs_mort)))
    print("AUCPR:{0:0.3f}".format(np.mean(aucprs_mort)))
    print("MCC: {0:0.3f}".format(np.mean(mccs_mort)))
    print("Spec@90: {0:0.3f}".format(np.mean(specat90_mort)))

    return {'Mean AUC': mean_auc_mort,
            'STD':std_auc_mort,
            'PPV':np.mean(ppvs_mort),
            'NPV':np.mean(npvs_mort),
            'AUCPR':np.mean(aucprs_mort),
            'MCC':np.mean(mccs_mort),
            'Spec@90':np.mean(specat90_mort)}        

#Phenotyping
def train_phen(config):
    from data_extraction.utils import normalize_data_phe as normalize_data
    from data_extraction.data_extraction_phenotyping \
            import data_extraction_phenotyping
    df_data, df_label = data_extraction_phenotyping(config)
    df = df_data.merge(df_label.drop(columns=['itemoffset']),
            on='patientunitstayid')
    import pdb;pdb.set_trace()
    all_idx = np.array(list(df['patientunitstayid'].unique()))   
    phen_auc  = []
    phen_aucs = []
    skf = KFold(n_splits=config.k_fold)
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(all_idx)):
        print('Running Fold {}...'.format(fold_id+1))
        train_idx = all_idx[train_idx]
        test_idx = all_idx[test_idx]

        train, test = normalize_data(config, df,train_idx, test_idx)
        train_gen, train_steps, (X_test, Y_test), max_time_step_test \
                = data_reader.read_data(config, train, test, val=False)
     
        model = network(config, 200, output_dim=25, activation='sigmoid')
        history = model.fit_generator(train_gen,steps_per_epoch=25,
                            epochs=config.epochs,verbose=1,shuffle=True)
    

        if config.num and config.cat:
            if config.ohe:
                x_cat = X_test[:, :, :7].astype(int)
                x_nc = X_test[:,:,7:]
                print("Please wait, One-hot encoding ...")
                one_hot = np.zeros((x_cat.shape[0], x_cat.shape[1], 429),
                        dtype=np.int)
                x_cat = (np.eye(config.n_cat_class)[x_cat].sum(2) > 0)\
                        .astype(int)
                probas_phen = model.predict([x_nc, x_cat])
                #todo Replace np.eye with faster function
            else:
                probas_phen = model.predict([X_test[:,:,7:],X_test[:,:,:7]])
        elif config.cat:
            if config.ohe:
                x_cat = X_test[:, :, :].astype(int)
                print("Please wait, One-hot encoding ...")
                one_hot = np.zeros((x_cat.shape[0], x_cat.shape[1], 429),
                        dtype=np.int)
                x_cat = (np.eye(config.n_cat_class)[x_cat].sum(2) > 0)\
                        .astype(int)
                probas_phen = model.predict([x_cat])
                #todo Replace np.eye with faster function
            else:
                probas_phen = model.predict([X_test])
        else:
            probas_phen = model.predict([X_test])   

        phen_auc = evaluation.multi_label_metrics(Y_test,probas_phen)
        phen_aucs.append(phen_auc)
    aucs_mean = np.mean(np.array(phen_aucs),axis=0)
    aucs_std  =  np.std(np.array(phen_aucs),axis=0)
    for i in range(len(config.col_phe)):
        print("{0} : {1:0.3f} +- {2:0.3f}".format(config.col_phe[i],
            aucs_mean[i],aucs_std[i]))
    return {'AUROC mean': aucs_mean,
            'AUROC std': aucs_std}

# Remaining length of stay
def train_rlos(config):
    from data_extraction.utils import normalize_data_rlos as normalize_data
    from data_extraction.data_extraction_rlos import data_extraction_rlos
    df_data = data_extraction_rlos(config)
    all_idx = np.array(list(df_data['patientunitstayid'].unique()))

    log_dir = pj( HOME, "tb-logs", g_session_str )
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
    models_dir = pj(config.output_dir, g_session_str, "models")
    os.makedirs(models_dir)

    r2s= []
    mses = []
    maes = []
    skf = KFold(n_splits=config.k_fold)
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(all_idx)):
        print('Running Fold {}...'.format(fold_id+1))
        train_idx = all_idx[train_idx]
        test_idx = all_idx[test_idx]

        train, test = normalize_data(config, df_data,train_idx, test_idx)
        train_gen, train_steps, (X_test, Y_test), max_time_step_test \
                = data_reader.read_data(config, train, test, val=False)
      
        model = network(config, 200, output_dim=1, activation='relu')
        write_summary( pj(config.output_dir, g_session_str), model)

        prefix = f"fold_{fold_id}"
        checkpoint_path = pj(models_dir, prefix + "_cp-{epoch:04d}.ckpt")
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=1, 
            save_weights_only=True,
            save_freq=config.save_freq)

        history = model.fit(train_gen, steps_per_epoch=25,
                            epochs=config.epochs, verbose=1, shuffle=True,
                            callbacks=[tensorboard_callback, cp_callback])
        if config.num and config.cat:
            if config.ohe:
                x_cat = X_test[:, :, :7].astype(int)
                x_nc = X_test[:,:,7:]
                print("Please wait, One-hot encoding ...")
                one_hot = np.zeros((x_cat.shape[0], x_cat.shape[1], 429),
                        dtype=np.int)
                x_cat = (np.eye(config.n_cat_class)[x_cat].sum(2) > 0)\
                        .astype(int)
                probas_rlos = model.predict([x_nc, x_cat])
                #todo Replace np.eye with faster function
            else:
                probas_rlos = model.predict([X_test[:,:,7:],X_test[:,:,:7]])
        elif config.cat:
            if config.ohe:
                x_cat = X_test[:, :, :].astype(int)
                print("Please wait, One-hot encoding ...")
                one_hot = np.zeros((x_cat.shape[0], x_cat.shape[1], 429),
                        dtype=np.int)
                x_cat = (np.eye(config.n_cat_class)[x_cat].sum(2) > 0)\
                        .astype(int)
                probas_rlos = model.predict([x_cat])
                #todo Replace np.eye with faster function
            else:
                probas_rlos = model.predict([X_test])
        else:
            probas_rlos = model.predict([X_test])   

        r2,mse,mae = evaluation.regression_metrics(Y_test,probas_rlos,
                max_time_step_test)
        r2s.append(r2)
        mses.append(mse)
        maes.append(mae)

    meanr2s = np.mean(r2s)
    meanmses = np.mean(mses)
    meanmaes = np.mean(maes)

    stdr2s = np.std(r2s)
    stdmses = np.std(mses)
    stdmaes = np.std(maes)


    print("===========================RLOS=============================")
    print("R2 total: {0:0.3f} +- {1:0.3f} ".format(meanr2s,stdr2s))
    print("MSE total: {0:0.3f}  +- {1:0.3f}".format(meanmses,stdmses))
    print("MAE total:{0:0.3f}  +- {1:0.3f}".format(meanmaes,stdmaes))

    return {'R2 mean': meanr2s,
         'R2 std': stdr2s,
         'MSE mean':meanmses,
         'MSE std':stdmses,
         'MAE mean':meanmaes,
         'MAE std':stdmaes}


def main(config):
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 1
    session = tf.compat.v1.Session(config=tf_config)
    K.set_session(session)

    set_session_str(config.task)
    config.write(g_session_str)
    if not pe(config.output_dir):
        os.makedirs(config.output_dir)

    np.random.seed(config.seed)

    if config.task == 'dec':
        result = train_dec(config)
    elif config.task =='mort':
        result = train_mort(config)
    elif config.task == 'phen':
        result = train_phen(config)
    elif config.task =='rlos':
        result = train_rlos(config)
    elif config.task == "dpsom_mort":
        result = train_dpsom_mort(config)
    else:
        print('Invalid task name')

    output_file_name = 'LSTM_{}_{}_{}_{}_{}_{}.json'.format(config.task,
            str(config.num), str(config.cat), str(config.ann), str(config.ohe),
            config.mort_window)
    output_file_path = pj(config.output_dir, output_file_name)
    with open(output_file_path, 'w') as f:
        f.write(str(result))
    print(f"Results written to {output_file_path}")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default='mort', type=str, required=False)
    parser.add_argument("--num", default=True, type=str, required=False)
    parser.add_argument("--cat", default=True, type=str, required=False)
    parser.add_argument("--ann", default=False, type=str, required=False)
    parser.add_argument("--ohe", default=False, type=str, required=False)
    parser.add_argument("--mort_window", default=48, type=int, required=False)

    parser.add_argument("-b", "--batch-size", type=int, default=512)
    parser.add_argument("-e", "--epochs", type=int, default=100)

    args = parser.parse_args()
    config = Config(args)
    main(config)
