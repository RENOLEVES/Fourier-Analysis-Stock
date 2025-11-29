from model_package import ModelTools
import pandas as pd


class Feature_Reduction(ModelTools):

    def feature_reduction(self, func, data, percent=0.01, top=50):
        self.feature_importance = self.get_impt(func, data)
        self.top_features = self.select_top(self.feature_importance, percent, top)

    def correlation_reduction(self, data, thres=0.6):
        self.corr_matrix = data.corr()
        self.corr_matrix_reduced = self.corr_matrix.map(lambda x: self.corr_filter(x, thres))
        self.best_cols = self.get_best_cols(self.corr_matrix_reduced)
        self.best_features = self.top_features[self.top_features['Name'].isin(self.best_cols)]

    @staticmethod
    def get_impt(f, d):
        impt = pd.DataFrame()
        impt['Name'] = d.columns
        impt['Feature_Importance'] = f.feature_importances_
        impt = impt.sort_values(['Feature_Importance'], ascending=False)
        return impt

    @staticmethod
    def select_top(impt, percent=0.01, top=50):
        impt_thres = (impt['Feature_Importance'].iloc[0]) * percent
        impt_per = impt[impt['Feature_Importance'] >= impt_thres]
        impt_top = impt.head(top)
        if len(impt_per) <= top:
            return impt_per
        else:
            return impt_top

    @staticmethod
    def corr_filter(x, thres=0.6):
        if x > thres or x < -thres:
            return 1
        else:
            return 0

    @staticmethod
    def get_best_cols(dat):
        if dat.empty:
            return []
        else:
            col_name = dat.index[0]
            del_ind = dat.index[dat[col_name] == 1]
            dat = dat[dat[col_name] == 0]
            for col in del_ind:
                del dat[col]
            return [col_name] + Feature_Reduction.get_best_cols(dat)
