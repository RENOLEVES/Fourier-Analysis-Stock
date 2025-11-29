from segmentation import *


class Backtesting(Segmentation):

    def backtest(self, data, tar_col, id_col=None, index_col=None, real_col=None, segment="y", graph="n", output="n"):

        # check if model is trained
        if self.model_algo is None:
            raise ValueError("Model is not trained.")
        # check data is a dataframe format
        if not isinstance(data, pd.Dataframe):
            raise ValueError("Currency data has to be in dataframe format")
        # check if our selected features are in the given data
        if not set(self.best_cols).issubset(set(data.columns)):
            raise ValueError("Selected features do not match columns in backtesting data!")
        # check target_column exist
        if tar_col not in data.columns:
            raise ValueError("Target column do not exist in data")

        key_cols = []
        if id_col == None:
            warnings.warn("No Id column given, will not be able to provide (key, segment) pair", UserWarning)
        elif id_col not in data.columns:
            raise ValueError("Id column do not exist in data")

        else:
            key_cols.append(id_col)
        # check if realized value column exist
        if real_col == None:
            warnings.warn("No realized values given, will not compute segment averages", UserWarning)
        elif real_col not in data.columns:
            raise ValueError("Realized values column do not exist in data")
        else:
            key_cols.append(real_col)
        # check if index column exist
        if index_col == None:
            warnings.warn("No index column given, will not be able to provide plot over index periods", UserWarning)
        elif index_col not in data.columns:
            raise ValueError("Index column do not exist in data")
        else:
            key_cols.append(index_col)

        # check if we have done segmentation already
        if self.segment == "n" and segment == "y":
            raise ValueError("Segmentation not done, please double check, or set segment = 'n")

        self.do_backtest(data, tar_col, index_col, real_col, key_cols, segment)

        if graph == "y":
            if index_col == None:
                raise ValueError("No index column given, will not be able to provide plot over index periods")
            else:
                self.backtest_graph(index_col)

        if output == "y":
            if os.path.isfile("backtest_result_" + str(self.version) + ".pkl"):
                ModelTools.overwrite_warning("backtest_result_")
            joblib.dump(self.bt_result, "backtest_result_" + str(self.version) + ".pkl")


    def do_backtest(self, data, tar_col, index_col, real_col, key_cols, segment):
        # impute input data
        if self.imputer == None:
            raise AttributeError("No imputer found, potential error in modelling process. Please double check", UserWarning)
        # if no index_col, this is a one time backtest
        if index_col == None:
            self.bt_result = self.single_test(data[key_cols], data[self.best_cols], data[tar_col], real_col, segment)
        else:
            # otherwise we have multiple backtest periods
            self.bt_result = pd.Dataframe()
            for key in data[index_col].unique():
                data_s = data[data[index_col] == key]
                data_s.reset_index(drop=True, inplace=True)
                s_res = self.single_test(data_s[key_cols], data_s[self.best_cols], data_s[tar_col], real_col, segment)
                self.bt_result = self.bt_result.append(s_res)


    def single_test(self, key_data, input_data, target, real_col, segment):
        # impute input
        if self.imputer != None:
            imputed_data = self.imputer.transform(input_data)
        else:
            warnings.warn("No imputer given, no imputation is performed", UserWarning)
            imputed_data = input_data

        key_data["Perdiction"] = self.get_prediction(imputed_data)

        if segment == 'y':
            key_data['Segment'] = pd.cut(key_data["Prediction"], self.bins, labels=np.arange(1, len(self.bins)))
            if real_col != None:
                key_data['Seg_Mean'] = key_data.groupby('segment')[real_col].transform('mean')

        key_data['Score'] = self.get_score(imputed_data, target, key_data["Prediction"])
        return key_data


    def get_prediction(self, input_data):
        if self.model_type == 'regressor':
            return pd.Series(self.rf.predict(input_data))
        else:
            return pd.Series(self.rf.predict_proba(input_data)[:, 1])
