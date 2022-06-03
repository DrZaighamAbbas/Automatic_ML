# Import libraries
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import category_encoders as ce
import timeit
import category_encoders
import os
from math import sqrt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder, normalize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn_pandas import CategoricalImputer
import xgboost as xgb
from tpot import TPOTRegressor
from sklearn.model_selection import check_cv
import autosklearn.classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,precision_recall_fscore_support
import pickle

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 300)

import logging
from datetime import datetime
from utils.file_writer import *
class Auto_Classification(object):
    def __init__(self):
        logging.info("AutoML Classification")
        pass

    def run(self,config,data_frame):
        logging.info("Run Auto_SKlearn Classifier")
        df=pd.DataFrame(data_frame)
        print(df.head())
        # break into X and y dataframes
        X = df.reindex(columns=[x for x in df.columns.values if x != 'Class'])  # separate out X

        y = df.reindex(columns=['electric_power'])

        y = np.ravel(y)  # flatten the y array

        # make list of numeric and string columns
        numeric_cols = []  # could still have ordinal data
        string_cols = []  # could have ordinal or nominal data

        for col in X.columns:
            if (X.dtypes[col] == np.int64 or X.dtypes[col] == np.int32 or X.dtypes[col] == np.float32 or X.dtypes[
                col] == np.float64):
                numeric_cols.append(col)  # True integer or float columns

            if (X.dtypes[col] == np.object):  # Nominal and ordinal columns
                string_cols.append(col)

        # In[6]

        print(X[string_cols].head(2))
        print(X[numeric_cols].head(2))

        X_string = X[string_cols]
        X_string = X_string.fillna("missing")
        # imputing missing values with most freqent values for numeric columns
        imp = SimpleImputer(missing_values=np.nan, copy=True, strategy='most_frequent')
        # imputing with most frequent because some of these numeric columns are ordinal
        X_numeric = X[numeric_cols]
        X_numeric = imp.fit_transform(X_numeric)
        X_numeric = pd.DataFrame(X_numeric, columns=numeric_cols)
        # encode the X columns string values as integers
        X_string = X_string.apply(LabelEncoder().fit_transform)
        print(X.head(2))
        # add the string and numeric dataframes back together
        # X = pd.concat([X_numeric, X_string], axis=1, join_axes=[X_numeric.index])
        full = pd.concat([X_numeric, X_string], axis=1).reindex(X_numeric.index)
        print(full.shape)
        X = full.drop(['electric_power'], axis=1)
        y = full['electric_power']
        print(X.shape)
        print(np.isnan(X).any())
        X = X.values

        feature_types = (['numerical'] * 3) + ['categorical'] + (['numerical'] * 9)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20)
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=120,
            per_run_time_limit=30,

        )
        times = []
        scores = []

        # run 2 iterations
        for x in range(1):
         start_time = timeit.default_timer()
        automl.fit(X_train, y_train)
        automl.score(X_test, y_test)
        elapsed = timeit.default_timer() - start_time
        times.append(elapsed)

        scores.append(automl.score(X_test, y_test))

        # output results
        times = [time / 60 for time in times]
        print('Times:', times)
        print('Scores:', scores)


        print(automl.show_models())
        predictions = automl.predict(X_test)
        print("Accuracy: %.2f" % accuracy_score(y_test,predictions))
        print("F1-Score: %.2f" % f1_score(y_test,predictions, average='weighted'))
        print("Percision: %.2f" % precision_score(y_test,predictions, average='weighted'))
        print("Recall: %.2f" % recall_score(y_test, predictions, average='weighted'))
        print("Support:")
        print(precision_recall_fscore_support(y_test,predictions, average='weighted'))
        filename = 'data_output/model.pkl'
        pickle.dump(automl, open(filename, 'wb'))

        # some time later...

        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.score(X_test, y_test)
        print("Model_result:", result)












