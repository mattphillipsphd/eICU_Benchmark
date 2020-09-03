"""
Run inference and analytics on mortality predictions
"""

import argparse
import os


def main(cfg):
    model = load_model()
    probas_mort = model.predict([X_test[:,:,7:],X_test[:,:,:7]])

    from data_extraction.data_extraction_mortality \
            import data_extraction_mortality
    from data_extraction.utils import normalize_data_mort as normalize_data

    df_data = data_extraction_mortality(config)
    all_idx = np.array(list(df_data['patientunitstayid'].unique()))
    skf = KFold(n_splits=config.k_fold)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cfg = vars( parser.parse_args() )
    main(cfg)

