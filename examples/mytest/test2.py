
import numpy as np
import statsmodels.api as sm

#You can also use numpy arrays instead of formulas:
nobs = 100

X = np.random.random((nobs,2))
X = sm.add_constant(X)
beta = [1, .1, .5]
e = np.random.random(nobs)
y = np.dot(X, beta) + e
results = sm.OLS(y, X).fit()
print(results.summary())