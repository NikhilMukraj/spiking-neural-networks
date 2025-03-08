#   - Define target behavior
#     - Probably as a Python function
#   - Assume that trends are linear and will continue in that direction
#   - Track which changes contribute to which trends
#   - Go in direction of desired target
# Start with Bayesian inference pipeline

# import numpy as np
# from sklearn.linear_model import LinearRegression
# from scipy.stats import pearsonr


# key is parameters, value is score
# when multiple values are found, find linear trend
# find the most correlated linear trend and move in the direction of closer to target
analysis = {}


# features are likely correlated with one another
# may need polynomial regression that takes in all features and 
# use that polynomial equation to determine direction to take
# def return_new_parameters(analysis, target, step_size):
#     trends = {}
#     for key, value in analysis:
#         for n, i in enumerate(key):
#             if n not in trends:
#                 trends[n] = {}
#             trends[n][i] = value

#     trends_analysis = {}

#     for i in trends:
#         X, Y = np.array(i.keys()), np.array(i.values())
#         reg = LinearRegression.fit(X, Y)
#         r = pearsonr(X, Y)
#         m = reg.coef_[0]
#         interecept = reg.intercept_

#         trends_analysis[i] = {'r' : r, 'm' : m, 'intercept': intercept}

