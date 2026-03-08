Builds decision trees by splitting on linear combinations of randomly
chosen variables. Projection Pursuit (PP) is used to choose a projection
of the variables that best separates the classes. Using linear
combinations of variables to separate classes takes the correlation
between variables into account which allows PPTree to outperform a
traditional decision tree when separations between groups occurs in
combinations of variables. PPTree models can be assembled into PPForest
models, which correspond to the random forest version of the PPTree
model.
