from model_package import *


class Sampling(ModelTools):

    def sample(self, method='random', percent=0.1, key_col=None, time_col=None, num_samples=1, remove_na=None,
               rand_seed=None):
        if self.sampled == 'y':
            warnings.warn("Data already sampled, re-sampling not recommended", UserWarning)

        if remove_na is not None:
            if isinstance(remove_na, (int, float)) == False:
                raise ValueError("remove_na must be numeric type")
            elif ((remove_na < 0) | (remove_na > 1)):
                raise ValueError("remove_na must be percentage value between 0 and 1")
            else:
                self.sample_remove_na(remove_na)

        if rand_seed is None:
            rand_seed = self.seed

        if method == 'random':
            if isinstance(percent, (int, float)) == False:
                raise ValueError("percent must be numeric type")
            elif ((percent < 0) | (percent > 1)):
                raise ValueError("Percentage must be percentage value between 0 and 1")
            else:
                self.random_sample(percent, rand_seed)

        if method == 'by_key':
            if not key_col in self.data.columns:
                raise ValueError("key_col must exist in training data for by_key method")
            if isinstance(num_samples, (int, float)) == False:
                raise ValueError("num_samples must be numeric type")
            else:
                self.per_key_sample(key_col, num_samples, rand_seed)

        if method == 'by_key_time':
            if not key_col in self.data.columns:
                raise ValueError("key_col must exist in training data for by_key method")
            if not time_col in self.data.columns:
                raise ValueError("time_col must exist in training data for by_key method")
            if isinstance(num_samples, (int, float)) == False:
                raise ValueError("num_samples must be numeric type")
            else:
                self.per_key_time_sample(key_col, time_col, num_samples, rand_seed)

        self.get_sample_tar()

    def sample_remove_na(self, remove_na):

        num_cols = len(self.data.columns)
        mask = self.data.isnull().sum(axis=1) / num_cols <= remove_na
        self.data_s_na = self.data[mask]
        self.sample_na = 'y'

    def random_sample(self, percent, rand_seed=None):
        if self.sample_na == 'y':
            self.sampled_data = self.data_s_na.reset_index().sample(frac=percent, random_state=rand_seed)
        else:
            self.sampled_data = self.data.reset_index().sample(frac=percent, random_state=rand_seed)
        self.sampled = 'y'

    def per_key_sample(self, key_col, num_samples, rand_seed=None):
        np.random.seed(rand_seed)
        sample_func = lambda grp: grp.loc[np.random.choice(grp.index, min(len(grp), num_samples), False), :]
        if self.sample_na == 'y':
            self.sampled_data = self.data_s_na.reset_index().groupby(key_col, as_index=False).apply(sample_func)
        else:
            self.sampled_data = self.data.reset_index().groupby(key_col, as_index=False).apply(sample_func)
        self.sampled = 'y'

    def per_key_time_sample(self, key_col, time_col, num_samples, rand_seed=None):
        np.random.seed(rand_seed)
        sample_func = lambda grp: grp.loc[np.random.choice(grp.index, min(len(grp), num_samples), False), :]
        if self.sample_na == 'y':
            key_df = pd.DataFrame({key_col: self.data_s_na[key_col].unique()})
            time_df = pd.DataFrame({time_col: self.data_s_na[time_col].unique()})
        else:
            key_df = pd.DataFrame({key_col: self.data[key_col].unique()})
            time_df = pd.DataFrame({time_col: self.data[time_col].unique()})

        key_df['Join'] = 1
        time_df['Join'] = 1

        merged_df = pd.merge(key_df, time_df, on='Join')
        merged_df.drop('Join', axis=1)
        sampled_df = merged_df.groupby(key_col, as_index=False).apply(sample_func)

        if self.sample_na == 'y':
            self.sampled_data = pd.merge(self.data_s_na.reset_index(), sampled_df, how='inner', on=[key_col, time_col])
        else:
            self.sampled_data = pd.merge(self.data.reset_index(), sampled_df, how='inner', on=[key_col, time_col])
        self.sampled = 'y'

    def get_sample_tar(self):
        self.sampled_target = self.target.filter(self.sampled_data['index'])
        self.sampled_data.drop('index', axis=1)
