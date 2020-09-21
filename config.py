import json
import os

pe = os.path.exists
pj = os.path.join
HOME = os.path.expanduser("~")

class Config():
    def __init__(self, args):
        self.seed = 36

        # data dir
        self.root_dir = pj( HOME, "Datasets/EHRs/eICU/eICU_benchmark" )
        self.eicu_dir = pj( HOME, "Datasets/EHRs/eICU/csv" )
        self.output_dir = pj( HOME, "Training/eICU_benchmark" )

        # Data prepared a la the DPSOM paper
        self.dpsom_supdir = pj( HOME, "Datasets/EHRs/eICU" )
        self.dpsom_input_dim = 98
        self.dpsom_time_dim = 72 # Last N time steps
        self.test_pct = 0.2

        # Training
        self.num_workers = 1

        # task details
        self.task = args.task if hasattr(args, "task") else "mort"
            #['phen', 'dec', 'mort', 'rlos']
        self.num = args.num if hasattr(args, "num") else True
        self.cat = args.cat if hasattr(args, "cat") else True
        self.n_cat_class = 429        

        self.k_fold = 5
        #model params
        self.model_dir = ''
        self.embedding_dim = 5
        self.epochs = args.epochs if hasattr(args, "epochs") else 100
        self.batch_size = args.batch_size if hasattr(args, "batch_size") \
                else 512
        self.save_freq = 500

        self.ann = args.ann if hasattr(args, "ann") else False
        self.ohe = args.ohe if hasattr(args, "ohe") else False
        self.mort_window = args.mort_window if hasattr(args, "mort_window") \
                else 48
        self.lr = 0.0001
        self.dropout = 0.3
        self.rnn_layers = 2
        self.rnn_units = [64, 64]


        # decompensation
        self.dec_cat = ['apacheadmissiondx', 'ethnicity', 'gender',
                'GCS Total', 'Eyes', 'Motor', 'Verbal']
        self.dec_num = ['admissionheight', 'admissionweight', 'age',
                'Heart Rate', 'MAP (mmHg)','Invasive BP Diastolic',
                'Invasive BP Systolic', 'O2 Saturation', 'Respiratory Rate',
                'Temperature (C)', 'glucose', 'FiO2', 'pH']

        #phenotyping
        self.col_phe = ["Respiratory failure", "Fluid disorders",
                "Septicemia", "Acute and unspecified renal failure",
                "Pneumonia", "Acute cerebrovascular disease",
                "Acute myocardial infarction", "Gastrointestinal hem", "Shock",
                "Pleurisy", "lower respiratory", "Complications of surgical",
                "upper respiratory", "Hypertension with complications",
                "Essential hypertension", "CKD", "COPD", "lipid disorder",
                "Coronary athe", "DM without complication",
                "Cardiac dysrhythmias", "CHF", "DM with complications",
                "Other liver diseases", "Conduction disorders"]

    def write(self, session_str):
        """
        Write all parameter values to pj(self.output_dir, session_str)
        """

        config_dir = pj(self.output_dir, session_str)
        if not pe(config_dir):
            os.makedirs(config_dir)
        config_path =  pj(config_dir, "config.json")
        json.dump(self.__dict__, open(config_path, "w"), indent=4)

