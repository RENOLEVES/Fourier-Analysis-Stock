from model_package import *


class Scoring_Functions(ModelTools):

    def get_scoring_func(self, scoring):
        if scoring is None:
            if self.model_algo == 'Random Forest Regressor':
                self.scoring_type = 'spearman_score'
            elif self.model_algo == 'Random Forest Classifier':
                self.scoring_type = 'accuracy_score'
            else:
                self.scoring_type = 'Default'

            self.scoring_name = None
            return None, None
        else:
            score = getattr(self, scoring)
            self.scoring_type = scoring
            self.scoring_name = scoring
            return score()

    def auc_roc(self):
        return make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True), roc_auc_score

    def spearman(self):
        return make_scorer(self.spearman_func), self.spearman_func

    @staticmethod
    def spearman_corr(gini_input, y_r='y_r', y_p='y_p'):
        """
        Calculate a normalized Gini-like coefficient based on Spearman rank correlation.

        Parameters:
        gini_input (DataFrame): DataFrame containing actual values (y_r) and predicted values (y_p).
        y_r (str): Column name for actual values.
        y_p (str): Column name for predicted values.

        Returns:
        float: Spearman rank correlation coefficient between actual and predicted values.
        """
        from scipy.stats import spearmanr

        # Calculate Spearman rank correlation between actual and predicted
        corr, _ = spearmanr(gini_input[y_r], gini_input[y_p])

        return corr

    def spearman_func(self, y, y_pred):
        gini_input = pd.DataFrame({'y_r': y, "y_p": y_pred})
        return self.spearman_corr(gini_input)
