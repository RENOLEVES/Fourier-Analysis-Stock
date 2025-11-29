from model_package import *


class Preprocessing(ModelTools):

    def preprocessing(self, remove_na=None, infer_cat=None, impute="mean"):
        if self.preprocessed == 'y':
            raise ValueError("Already preprocessed, cannot do it twice")
        if infer_cat is not None:
            if not isinstance(infer_cat, (int, float)):
                raise ValueError("'infer_cat' must be numeric type.")
            elif (infer_cat < 0) | (infer_cat > 1):
                raise ValueError("'infer_cat' is either None or a percentage value between 0 and 1.")
            else:
                if self.cat_cols:
                    print("User already specified some categorical columns, shall try to append")
                if infer_cat > 0:
                    print("Warning: User chose to infer a percentage of numeric features as categorical")
                self.infer_cat(infer_cat)

        if remove_na is not None:
            if not isinstance(remove_na, (int, float)):
                raise ValueError("'remove_na' must be numeric type.")
            elif (remove_na < 0) | (remove_na > 1):
                raise ValueError("'remove_na' is either None or a percentage value between 0 and 1.")
            else:
                self.remove_na(remove_na)

        if self.valid == 'y':
            self.data_all = self.data.append(self.valid_data)
            self.data_all.rest_index(inplace=True, drop=True)
        else:
            self.data_all = self.data

        self.data_all = pd.get_dummies(self.data_all, dummy_na=True, columns=self.cat_cols)
        if self.valid == 'y':
            self.data = self.data_all.head(len(self.data))
            self.valid_data = self.data_all.tail(len(self.valid_data))
        else:
            self.data = self.data_all
        self.update_data_param()

        if self.valid == 'y':
            self.x_train = self.data
            self.x_valid = self.valid_data
            self.y_train = self.target
            self.y_valid = self.valid_target
        else:
            self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.data, self.target,
                                                                                      test_size=0.33,
                                                                                      random_state=self.seed)

        self.impute = impute
        imp = SimpleImputer(strategy=self.impute)
        imp.fit(self.x_train)

        self.x_train = pd.DataFrame(imp.transform(self.x_train))
        self.x_train.columns = self.data.columns
        self.x_valid = pd.DataFrame(imp.transform(self.x_train))
        self.x_valid.columns = self.data.columns

        self.preprocessed = 'y'

    def infer_cat(self, percent):
        non_numeric = list(self.data.select_dtypes(include=['object', 'category']).columns.values)
        numeric = list(self.data.select_dtypes(include=[np.number]).columns.values)
        if percent == 0:
            self.cat_cols = list(set().union(self.cat_cols, non_numeric))
        else:
            thres = self.len * percent
            unique_stats = self.data[numeric].nunique()
            unique_filtered = list(unique_stats[unique_stats < thres].index)
            self.cat_cols = list(set().union(self.cat_cols, non_numeric, unique_filtered))

    def remove_na(self, remove_percent):
        col_na = pd.DataFrame(columns=['Name', 'NA_Count'])
        i = 0
        for col in self.data.columns:
            col_na.loc[i] = [col, self.data[col].isnull().sum()]
            i = i + 1
        col_na['NA_Ratio'] = col_na["NA_Count"] / self.len
        no_na_cols = col_na["Name"][col_na["NA_Ratio"] <= remove_percent]
        na_cols = col_na["Name"][col_na["NA_Ratio"] > remove_percent]

        self.na_stats = col_na
        self.removed_cols = na_cols
        self.data = self.data[no_na_cols]

        self.update_data_param()

        self.cat_cols = list(filter(lambda x: x in self.cat_cols, no_na_cols))
