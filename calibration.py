import os
import sys
import pandas as pd
import sys
sys.path.insert(0, r"E:\IntelliJ IDEA 2020.2\IdeaProjects\stock\package")
from modeller import Modeller

os.chdir(r"E:\IntelliJ IDEA 2020.2\IdeaProjects\stock\data\source")

data = pd.read_csv("cleaned_btc_2022-06_2025-06.csv")

os.chdir(r"E:\IntelliJ IDEA 2020.2\IdeaProjects\stock\data\model")

data.reset_index(drop=True, inplace=True)

target = data['score']

features = [
    'Close', 'Open', 'High', 'Low', 'Prev Close', "Volume",
    'p_change', 'atr14', 'atr21',
    'ma_diff', 'ma_ratio', 'ma_short','ma_long', 'volatility',
    'golden_cross', 'death_cross', 'Weekday'
]

cat_cols = ["golden_cross", "death_cross"]

model1 = Modeller(data[features], target, 'regressor', cat_cols=cat_cols)

model1.preprocessing()

model1.rf_manual(scoring='spearman', complexity="complex", reduce=0.01)

model1.version='v1'

model1.show()

model1.output(detailed='y')
