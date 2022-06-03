import h2o
from h2o.automl import H2OAutoML
h2o.init()
import pandas as pd
from sklearn.metrics import r2_score

import logging
from datetime import datetime
from utils.file_writer import *


class H2OClassification(object):
    def __init__(self):
        logging.info("AutoML created Classification")
        pass

    def run(self, config, data_frame):
        logging.info("Run H2o Classifier")

        pd.set_option('display.max_rows', 20)
        pd.set_option('display.max_columns', 300)

        df = pd.DataFrame(data_frame)

        df = h2o.H2OFrame(data_frame)




        print(df.head())

        print(df.describe())

        y = "weekend"
        x = df.columns
        x.remove(y)


        splits = df.split_frame(ratios=[0.80], seed=1)
        train = splits[0]
        test = splits[1]
        aml = H2OAutoML(max_models=10, seed=1)
        aml.train(x=x, y=y, training_frame=df)
        lb= aml.leaderboard
        print(lb.head(5))

        # Get model ids for all models in the AutoML Leaderboard
        model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:, 0])
        # Get the "All Models" Stacked Ensemble model
        se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels" in mid][0])
        # Get the Stacked Ensemble metalearner model
        metalearner = h2o.get_model(se.metalearner()['name'])
        metalearner.coef_norm()
        metalearner.std_coef_plot()

        print(metalearner)
        print(aml.leaderboard)


        h2o.save_model(aml.leader, path="data_output/model")
        aml.leader.download_mojo(path="data_output/mojo.zip")

       # test_data = h2o.import_file("data/Filpettrain.csv")
        #from h2o.estimators import H2OStackedEnsembleEstimator
        #original_model = H2OStackedEnsembleEstimator()
        #original_model.train(x=["tz", "electric_power"], y="electric_power", training_frame=test_data)

        #import tempfile
        #original_model_filename = tempfile.mkdtemp()
        #original_model_filename = original_model.download_mojo(original_model_filename)
        mojo_model = h2o.import_mojo("data_output/mojo.zip")
        predictions = mojo_model.predict(test)
        print(predictions)
        model = mojo_model.model_performance(test)

        print(model)


