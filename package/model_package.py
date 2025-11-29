import pandas as pd
import numpy as np
import warnings
import joblib
import os.path
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, roc_auc_score


class ModelTools:

    def __init__(self, data, target, model_type='regressor', cat_cols=[], validation=None, valid_target=None,
                 version=0):

        if (model_type != 'regressor') & (model_type != 'classifier'):
            raise ValueError("Model type is either 'regressor', 'classifier'")
        else:
            self.model_type = model_type

        if set(cat_cols).issubset(set(data.columns)):
            self.cat_cols = cat_cols
        else:
            raise ValueError("Categorical features names inconsistent with data feature names. Please double check.")
        if not isinstance(validation, pd.DataFrame):
            self.valid = 'n'
            warnings.warn("No validation data given, or format problem. Shall create validation data from given data",
                          UserWarning)
        else:
            if (not isinstance(valid_target, pd.DataFrame)) and (not isinstance(valid_target, pd.Series)):
                raise ValueError("Validation data is given, but validation target not given or format problem")
            elif not set(validation.columns).issubset(set(data.columns)):
                raise ValueError("Column mismatch between training data and validation data")
            else:
                self.valid = 'y'
                self.valid_data = validation
                self.valid_target = valid_target

        self.data = data
        self.target = target
        self.columns = data.columns
        self.len = data.shape[0]
        self.width = data.shape[1]
        self.initial_features = data.shape[1]
        self.impute = None
        self.imputer = None
        self.removed_cols = []
        self.model_algo = None
        self.sampled = 'n'
        self.sample_na = 'n'
        self.preprocessed = 'n'
        self.na_stats = 'User did not choose to remove N/A columns'
        self.reduced_num = 0

        self.x_train = None
        self.x_valid = None
        self.y_train = None
        self.y_valid = None
        self.prediction = None

        self.segmented = 'n'
        self.bins = None

        self.version = version

        self.seed = 42

        def __repr__(self):
            if self.preprocessed == 'n':
                return "Data has been loaded. Please proceed with preprocessing and model training."
            elif self.model_algo is None:
                return "Data has been loaded and preprocessed, but no model has been trained."
            else:
                return "A {} is used. {} socrer is used for evaluation. A score of {} is obtained".format(
                    self.model_algo, self.scoring_type, self.score)

    def show(self):
        if self.model_algo is None:
            raise Exception("Model has not been run")

        info = pd.DataFrame(columns=["Name", "Info"])
        info.loc[0] = ["Number of training records:", len(self.x_train)]
        info.loc[1] = ["Number of validation records:", len(self.x_valid)]
        info.loc[2] = ["Number of initial features:", self.initial_features]
        info.loc[3] = ["Number of removed features:", len(self.removed_cols)]
        info.loc[4] = ["Number of final features:", len(self.best_cols)]
        info.loc[5] = ["Number of features further reduced:", self.reduced_num]
        info.loc[6] = ["Type of model:", self.model_type]
        info.loc[7] = ["Model Complexity:", self.model_complexity]
        info.loc[8] = ["Modelling Algorithm:", self.model_algo]
        info.loc[9] = ["Performance metric:", self.scoring_type]
        info.loc[10] = ["Training Score:", self.score_t]
        info.loc[11] = ["Validation Score:", self.score]

        self.info = info
        return self.info

    def output(self, detailed='n'):
        if self.model_algo is None:
            raise Exception("Model has not been run.")

        self.show()
        self.info.to_csv("model_summary_" + str(self.version) + ".csv")

        joblib.dump(self.final_model, "final_model_" + str(self.version) + ".pkl")

        self.best_features.to_csv("model_best_features_" + str(self.version) + ".csv")
        pd.DataFrame(self.best_param, index=[0]).to_csv("model_best_params_" + str(self.version) + ".csv")

    def dump(self, name="model_output_complete"):
        joblib.dump(self, name + str(self.version) + ".pkl")

    def update_data_param(self):
        self.columns = self.data.columns
        self.len = self.data.shape[0]
        self.width = self.data.shape[1]
