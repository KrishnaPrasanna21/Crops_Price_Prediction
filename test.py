import pandas as pd
import random
import random

import pandas as pd
from dmba import plotDecisionTree
name = "static/Arhar.csv"
dataset = pd.read_csv("static/Arhar.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

        #from sklearn.model_selection import train_test_split
        #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

        # Fitting decision tree regression to dataset
from sklearn.tree import DecisionTreeRegressor
depth = random.randrange(7,18)
regressor = DecisionTreeRegressor(max_depth=depth)
regressor.fit(X, Y)
plotDecisionTree(regressor,feature_names=X.columns,rotate=True)