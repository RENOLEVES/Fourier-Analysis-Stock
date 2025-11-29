from rf_models import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree


class Segmentation(RF_Models):

    def segment(self, max_segs=8, min_samples=0.01, criterion=None, output="n", graph="n"):

        if self.model_algo is None:
            raise ValueError("Model not trained! Check if model has been built")

        if isinstance(max_segs, int) == False or max_segs < 2:
            raise ValueError("'max_segs parameter must be a positive integer greater than 1")
        elif isinstance(min_samples, float) == False or min_samples > 1 or min_samples < 0:
            raise ValueError("'min_sample' parameter nmust be a number between 0 and 1")
        elif max_segs * min_samples > 1:
            raise ValueError("Cannot satisfy both 'max seg' and 'min_sample' simultaneously")

        if max_segs > len(self.y_valid) / 10:
            warnings.warn("max_segs might be too big for the data available", UserWarning)

        self.get_seg(max_segs, min_samples, criterion)

        if output == 'y':
            if os.path.isfile("model_seg_bins_" + str(self.version) + ".pkl"):
                ModelTools.overwrite_warning("model_seg_bins_")
            joblib.dump(self.bins, "model_seg_bins_" + str(self.version) + ".pkl")

        if graph == 'y':
            self.get_tree()

    def get_seg(self, max_segs, min_samples, criterion):  # 1 usage
        # check for regressor or classifier
        if self.model_type == "regressor":
            if criterion not in ["squared_error", "absolute_error"]:
                warnings.warn("criterion must be 'mse' or 'mae', shall default to 'mse'", UserWarning)
                criterion = "squared_error"
            self.dt = DecisionTreeRegressor(criterion=criterion, max_leaf_nodes=max_segs, min_samples_leaf=min_samples)
        else:
            if criterion not in ["gini", "entropy"]:
                warnings.warn("criterion must be 'gini' or 'entropy'; shall default to 'gini'", UserWarning)
                criterion = "gini"
            self.dt = DecisionTreeClassifier(criterion=criterion, max_leaf_nodes=max_segs, min_samples_leaf=min_samples)

        self.dt.fit(self.prediction_t.reshape(-1, 1), self.y_train)

        # get the bin boundaries
        thres = pd.DataFrame(self.dt.tree_.threshold)
        thres.sort_values(by=0, inplace=True)
        thres = thres[thres[0] > 0]
        bins = list(thres[0])
        bins = [-0.000001] + bins + [float('inf')]
        self.bins = bins
        self.segmented = 'y'

    # ---
    # Function to output the decision tree, writes to a pdf
    # ---
    def get_tree(self):  # 1 usage
        import os
        import matplotlib.pyplot as plt
        from sklearn import tree

        plt.figure(figsize=(25, 15)) 
        

        tree.plot_tree(
            self.dt,
            feature_names=["P_val"],
            filled=True, 
            rounded=True,
            proportion=False,
            fontsize=10
        )
        
        name = "segmentation_tree_"
        file_name = name + str(self.version) + ".png" 
        
        try:
            plt.savefig(file_name)
            print(f"Decision Tree successfully saved to: {file_name}")
        except Exception as e:
            print(f"Error saving plot: {e}")
            
        plt.close()
