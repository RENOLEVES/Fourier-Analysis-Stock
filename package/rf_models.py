from preprocessing import *
from feature_reduction import *
from scoring import *
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, \
    GradientBoostingClassifier


class RF_Models(Preprocessing, Feature_Reduction, Scoring_Functions):  # 1 usage

    def rf_auto(self, scoring=None):
        self.rf_manual(scoring=scoring, complexity='moderate')

    def rf_manual(self, scoring=None, complexity='moderate', reduce=None):  # 4 usage (2 dynamic)
        self.rf_advanced(scoring=scoring, complexity=complexity, reduce=reduce, inpt_top=50)

    def rf_advanced(self, scoring=None, complexity='moderate', reduce=None, param_grid=None, corr_threshold=0.6,
                    inpt_perc=0.01, inpt_top=50, gb='n'):  # 1 usage
        # Check if data has been preprocessed.
        if self.preprocessed == 'n':
            print('Data has not been preprocessed. Shall attempt to train model anyway.')
            self.preprocessed = 'skipped'

        if self.valid == 'y':
            self.x_train = self.data
            self.x_valid = self.valid_data
            self.y_train = self.target
            self.y_valid = self.valid_target
        else:
            # Train / test split
            x_train, x_valid, self.y_train, self.y_valid = train_test_split(
                self.x_train, self.y_train, test_size=0.33, random_state=self.seed
            )
            self.x_train = pd.DataFrame(x_train)
            self.x_valid = pd.DataFrame(x_valid)
            self.x_train.columns = self.data.columns
            self.x_valid.columns = self.data.columns

        if gb == 'y':
            # update model_type
            if self.model_type == 'regressor':
                self.model_algo = 'Gradient Boosting Regressor'
            else:
                self.model_algo = 'Gradient Boosting Classifier'
        else:
            if self.model_type == 'regressor':
                self.model_algo = 'Random Forest Regressor'
            else:
                self.model_algo = 'Random Forest Classifier'

        # update model complexity
        self.model_complexity = complexity

        # get scoring function
        self.score, self.scorer = self.get_scoring_func(scoring)

        if (gb == 'y') and (complexity != 'manual'):
            raise ValueError("Please set complexity to 'manual' for gradientboost mode")

        # switch between 3 complexity cases:
        if complexity == 'basic':
            param_grid = {'max_depth': [5], 'n_estimators': [50], 'max_features': ['auto'], 'min_samples_leaf': [25]}
        elif complexity == 'moderate':
            param_grid = {'max_depth': [5, 10], 'n_estimators': [10, 50, 100], 'max_features': ['auto'],
                          'min_samples_leaf': [25, 50]}
        elif complexity == 'complex':
            param_grid = {'max_depth': [4, 6, 8, 10, None], 'n_estimators': [10, 50, 100, 150, 200],
                          'max_features': ['sqrt', 'log2', 1], 'min_samples_leaf': [25, 50, 75, 100]}
        elif complexity == 'manual':
            if param_grid is None:
                raise ValueError("Param_grid input needed for manual mode")
        else:
            raise ValueError("complexity takes values of 'basic', 'moderate', 'complex' or 'manual'")

        # run grid search and do all reductions
        self.rf_get_best(param_grid, corr_threshold, inpt_perc, inpt_top, gb)

        # get final model output
        self.rf_get_final_model()

        # add exhaustive test later

        # see if further reduction is necessary
        if reduce is not None:
            if isinstance(reduce, (int, float)) == False:
                raise ValueError("'reduce' must be numeric type")
            elif isinstance(reduce, int):
                if reduce < 0:
                    raise ValueError("Integer 'reduce' must be greater than 0")
                else:
                    self.rf_further_reduce_int(reduce)
            elif isinstance(reduce, float):
                if (reduce < 0) or (reduce > 1):
                    raise ValueError("Float 'reduce' must be between 0 and 1")
                else:
                    if reduce > 0.1:
                        warnings.warn("Performance drop of over 10% is not recommended", UserWarning)
                    self.rf_further_reduce_per(reduce)

        if self.impute != None:
            imp = SimpleImputer(strategy=self.impute)
            imp.fit(self.x_train[self.best_cols])
            self.imputer = imp

    def rf_grid_search(self, x_train, y_train, param_grid, cv=None, score=None, rand_state=None, gb='N'):
        if gb == 'y':
            if self.model_type == 'regressor':
                rf = GradientBoostingRegressor(warm_start=False, random_state=rand_state)
            else:
                rf = GradientBoostingClassifier(warm_start=False, random_state=rand_state)
        else:
            if self.model_type == 'regressor':
                rf = RandomForestRegressor(warm_start=False, oob_score=False, random_state=rand_state, n_jobs=-1)
            else:
                rf = RandomForestClassifier(warm_start=False, oob_score=False, random_state=rand_state, n_jobs=-1)

        grid_search = GridSearchCV(rf, param_grid=param_grid, scoring=score, cv=cv)
        grid_search.fit(x_train, y_train)
        clf_best = grid_search.best_estimator_
        cv_results = grid_search.cv_results_
        best_index = grid_search.best_index_
        return clf_best, cv_results, best_index

    def rf_get_best(self, param_grid, corr_threshold=0.6, impt_per=0.01, impt_top=50, gb='n'):
        self.rf, self.cv_results, self.best_index = self.rf_grid_search(self.x_train, self.y_train, param_grid,
                                                                        score=self.score, rand_state=self.seed, gb=gb)

        self.best_param = self.cv_results['params'][self.best_index]

        self.feature_reduction(self.rf, self.x_train, impt_per, impt_top)

        top_cols = self.top_features['Name']

        self.correlation_reduction(self.x_train[top_cols], corr_threshold)

    def rf_get_final_model(self):
        self.rf.fit(self.x_train[self.best_cols], self.y_train)
        self.final_model = self.rf

        if self.model_type == 'regressor':
            self.prediction = self.rf.predict(self.x_valid[self.best_cols])
            self.prediction_t = self.rf.predict(self.x_train[self.best_cols])

        else:
            self.prediction = self.rf.predict_proba(self.x_valid[self.best_cols])[:,1]
            self.prediction_t = self.rf.predict_proba(self.x_train[self.best_cols])[:,1]

        if self.scoring_name == None:
            self.score = self.rf.score(self.x_valid[self.best_cols], self.y_valid)
            self.score_t = self.rf.score(self.x_train[self.best_cols], self.y_train)

        else:
            self.score = self.scorer(self.y_valid, self.prediction)
            self.score_t = self.scorer(self.y_train, self.prediction_t)


    def rf_further_reduce_int(self, reduce):
        self.best_cols = self.best_cols[:reduce]
        self.rf_get_final_model()
        if reduce >= len(self.best_cols):
            self.reduced_num = len(self.best_cols)
        else:
            self.reduced_num = len(self.best_cols) - reduce


    def rf_further_reduce_per(self, reduce):
        max_score = self.score
        t_features = len(self.best_cols)
        l_features = t_features

        while(((abs(max_score - self.score)/max_score) < reduce) and (l_features > 1)):
            l_features = l_features - 1
            self.best_cols = self.best_cols[:l_features]
            self.rf_get_final_model()

        self.reduced_num = t_features - len(self.best_cols)
