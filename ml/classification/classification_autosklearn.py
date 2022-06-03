# Import libraries
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import category_encoders as ce
import timeit
import category_encoders
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import sklearn.metrics
import os
from sklearn.datasets import load_files

# Any results you write to the current directory are saved as output.
import timeit

from math import sqrt
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_boston
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder, normalize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn_pandas import CategoricalImputer
import xgboost as xgb
from tpot import TPOTClassifier
from sklearn.model_selection import check_cv
from sklearn.model_selection import train_test_split
from IPython import get_ipython
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,precision_recall_fscore_support, average_precision_score


pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 300)

import logging
from datetime import datetime
from utils.file_writer import *


class ClassificationASKL(object):
    def __init__(self):
        logging.info("AutoML created Classifier")
        pass

    def run(self, config, data_frame):
        logging.info("Run TPOT Classifier")

        pd.set_option('display.max_rows', 20)
        pd.set_option('display.max_columns', 300)
        df=pd.DataFrame(data_frame)
        print(df.head())
        print(df.info())
        print(df.describe())

        df = pd.DataFrame(data_frame)

        print(df.head())
        print(df.info())
        print(df.describe())

        # break into X and y dataframes
        # separate out X
        X = df.reindex(columns=[x for x in df.columns.values if x != 'Class'])

        y = df.reindex(columns=['electric_power'])  # target features

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

        # In[6]:

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
        print(X.head(10))
        # add the string and numeric dataframes back together
        # X = pd.concat([X_numeric, X_string], axis=1, join_axes=[X_numeric.index])
        full = pd.concat([X_numeric, X_string], axis=1).reindex(X_numeric.index)
        print(full.shape)
        print(np.isnan(full).any())
        print(full)

        # X.to_csv("dataset.csv")
        X = full.drop(['electric_power'], axis=1)
        y = full['electric_power']

        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20)

        # instantiate tpot
        tpot = TPOTClassifier(verbosity=2,

                              config_dict='TPOT light',

                              n_jobs=-4,
                              cv=5,
                              generations=3,

                              population_size=10,
                              )
        times = []
        scores = []
        winning_pipes = []

        for x in range(1):
            start_time = timeit.default_timer()
            tpot.fit(X_train, y_train)
            tpot.score(X_test, y_test)
            elapsed = timeit.default_timer() - start_time
            times.append(elapsed)
            winning_pipes.append(tpot.fitted_pipeline_)
            scores.append(tpot.score(X_test, y_test))
            tpot.export('tpot_Classification.py')
        times = [time / 60 for time in times]
        print('Times:', times)
        print('Scores:', scores)
        print('Winning pipelines:', winning_pipes)


        predictions = tpot.predict(X_test)
        print(predictions)

        # Create the submission file
        final = pd.DataFrame({'electric_power': predictions})

        final.to_csv('data_output/submission.csv', index=False)



        print("Accuracy: %.2f" % accuracy_score(y_test, predictions))
        print("F1-Score: %.2f" % f1_score(y_test,predictions, average='weighted'))
        print("Percision: %.2f" % precision_score(y_test,predictions,average='weighted'))
        print("Recall: %.2f" % recall_score(y_test,predictions, average='weighted'))
        print("Support:")
        print( precision_recall_fscore_support(y_test,predictions, average='weighted'))















