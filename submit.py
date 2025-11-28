import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your models using training CRPs
	# X_train has 8 columns containing the challenge bits
	# y_train contains the values for responses
	
	# THE RETURNED MODEL SHOULD BE ONE VECTOR AND ONE BIAS TERM
	# If you do not wish to use a bias term, set it to 0
	# Apply the feature map to the training data
	X_train_mapped = my_map(X_train)

	# Initialize and train Logistic Regression
	model = LogisticRegression(
    penalty='l2',        # Type of regularization ('l2' is default)
    C=1.0,               # Inverse of regularization strength; smaller means stronger regularization
    solver='liblinear',  # Solver for small datasets (or 'saga' for large)
    max_iter=10000
		)
	model.fit(X_train_mapped, y_train)

	# Initialize and train Ridge Regression
	# model = RidgeClassifier(
  #   alpha=1.0,         # Regularization strength; larger = more regularization
  #   solver='auto'      # Solver can be 'auto', 'svd', 'cholesky', etc.
	# 	)
	# model.fit(X_train_mapped, y_train)

	# # Initialize and train cSVM Regression
	# model = LinearSVC(
  #   penalty='l2',       # Type of regularization (only 'l2' supported)
  #   loss='hinge',  # Type of loss function
  #   C=0.001,              # Inverse of regularization strength
  #   max_iter=10000
	# 	)
	# model.fit(X_train_mapped, y_train)

	# Extract weights (W) and bias (b)
	w = model.coef_  # shape (1, 105) for binary classification
	b = model.intercept_  # shape (1,)
	return w, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
	m = X.shape[0]
	features = []

	for i in range(m):
			x = X[i]
			d = 1 - 2 * x.astype(int)  # convert to {-1, +1}

			# Build phi(c): first 8 values are d0 to d7
			phi = list(d)

			# Last 7 are cumulative products from d7 down to d1
			cumprod = 1
			for j in reversed(range(8)):
					cumprod *= d[j]
					if j < 7:
							phi.append(cumprod)

			phi = np.array(phi)  # shape (15,)

			#Compute all unique pairwise products (i < j)
			phi_cross = []
			for i in range(15):
					for j in range(i + 1, 15):
							phi_cross.append(phi[i] * phi[j])

			features.append(phi_cross)

	feat = np.array(features)  # shape (m, 105)
			
	return feat


################################
# Non Editable Region Starting #
################################
def my_decode( w ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to invert a PUF linear model to get back delays
	# w is a single 65-dim vector (last dimension being the bias term)
	# The output should be four 64-dimensional vectors

    # Construct sparse linear system A @ d = w
    A = np.zeros((65, 256))

    for i in range(65):
        if i < 64:
            A[i, i]     += 0.5   # d = p - q → α = (d + c)/2
            A[i, 64+i]  += -0.5
            A[i, 128+i] += 0.5   # c = r - s → β = (d - c)/2
            A[i, 192+i] += -0.5
        if i > 0:
            A[i, i-1]     += 0.5
            A[i, 64+i-1]  += -0.5
            A[i, 128+i-1] += -0.5
            A[i, 192+i-1] += 0.5

    # Solve least squares: minimize ||A @ d - w||^2
    d_hat, _, _, _ = np.linalg.lstsq(A, w, rcond=None)

    # Enforce non-negativity
    d_hat = np.maximum(d_hat, 0)

    # Split into p, q, r, s
    p, q, r, s = np.split(d_hat, 4)
    return p, q, r, s

