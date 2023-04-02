First I tried sklearn.ensemble.HistGradientBoostingRegressor, which is recommended by scikit-learn to handle NaN values without preprocessing. 
Then I tried to fill na with mean value, and used huber regressor. It turned out bad.
Then I tried sklearn.impute.IterativeImputer and KNNImputer with huber regressor. The result is not good enough.
Then I used Gaussian Regressor with some kernels.
