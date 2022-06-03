import sys
import logging

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from config.Config import Config

# Logging
root = logging.getLogger()
root.setLevel(logging.DEBUG)

config = None

def read_config():
    global config
    # Read Config file
    logging.info("Read configuration from %s" % config_file)
    config = Config.read_file(config_file)
    # For testing
    logging.debug("Configuration %s" % str(config))
    logging.debug("Problem type %s" % config['problem_type'])

def fetch_csv_dataset():
    input_dir = config['input_dir']
    from dataloader.csv_loader import load_data
    data_frame = load_data(input_dir)
    logging.info("data_frame from %s data_frame %s" % (input_dir, data_frame))
    logging.debug(str(data_frame.head()))
    return data_frame

def main(config_file):
    read_config()
    # get data
    data_frame = fetch_csv_dataset()
    problem_type = config['problem_type']
    if problem_type == "Classification":
        # Import classification logic
        from ml.classification.classification_autosklearn import ClassificationASKL
        ptype = ClassificationASKL()
        ptype.run(config, data_frame)
    elif problem_type == "Regression":
        # Import regression logic
        from ml.regression.regression_tpot import RegressionTPOT
        ptype = RegressionTPOT()
        ptype.run(config, data_frame)

    elif problem_type == "H2ORegression":
        # Import regression logic
        from ml.H2O_AUTOML_REGRESSION.H2O_REGRESSION import H2ORegression
        ptype = H2ORegression()
        ptype.run(config, data_frame)
    elif problem_type == "H2OClassification":
        # Import regression logic
        from ml.h2o_classification.h20_classifier import H2OClassification
        ptype = H2OClassification()
        ptype.run(config, data_frame)
    elif problem_type == "AutoSklearn_Classification":
            # Import classification logic
            from ml.Auto_SKL_Classification.ASK_Classification import Auto_Classification
            ptype = Auto_Classification()
            ptype.run(config,data_frame)
    else:
            # Import regression logic
            from ml.Auto_SKL_Regression.ASK_Regression import Auto_Regression
            ptype = Auto_Regression()
            ptype.run(config, data_frame)
    pass


if __name__ == '__main__':
    config_file = "./config.json"
    # if len(sys.argv) == 0:
    #     logging.error("Giv config file input after program name")
    #     #sys.exit(1)
    #     config_file = "./config.json"
    # else:
    #     config_file = sys.argv[1]

    main(config_file)