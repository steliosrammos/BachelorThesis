from ConformalBiasCorrection import ConformalBiasCorrection

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# Classifier S
from xgboost import XGBClassifier

import time

# Visualizations
# import sys
# sys.path.append('/Users/steliosrammos/Documents/Education/Maastricht/DKE-Year3/BachelorThesis/bachelor_thesis/src/visualization')
# import visualize

import warnings
warnings.filterwarnings('ignore')

# Import data
base_path = '/Users/steliosrammos/Documents/Education/Maastricht/DKE-Year3/BachelorThesis/bachelor_thesis/'

# Select dataset
train_data = pd.read_csv(base_path+"data/external/biased_train_ionosphere.csv", sep=";")
test_data = pd.read_csv(base_path+"data/external/test_ionosphere.csv", sep=";")

counts = train_data.got_go.value_counts()
ratio = counts[0]/counts[1]

X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

######### CLASSIFIER S #############
# XGBoost
classifier_s = XGBClassifier()
parameters_s = {}

######### CLASSIFIER y #############
# XGBoost
classifier_y = XGBClassifier()
parameters_y = {}

# Set classifiers and their parameters
classifiers = {
    'classifier_s': classifier_s,
    'classifier_y': classifier_y
}

clf_parameters = {
    'classifier_s': parameters_s,
    'classifier_y': parameters_y
}

# Set rebalancing and bias correction parameters
rebalancing_parameters = {
    'rebalancing_s': None,
    'SMOTE_y': False,
    'conformal_oversampling': False,
    'balance_new_labels': False}

bias_correction_parameters = {'correct_bias': True}

uncorrected_roc = []
corrected_roc = []

# Initialize some variables
sss = StratifiedKFold(n_splits=10)

uncorrected_rocs_y = []
corrected_rocs_y = []

rocs_s = []
briers_s = []

framework = ConformalBiasCorrection(train_data=train_data, test_data=test_data, classifiers=classifiers, clf_parameters=clf_parameters, rebalancing_parameters=rebalancing_parameters, bias_correction_parameters=bias_correction_parameters)
framework.verbose = 2
framework.random_state = 1

if bias_correction_parameters['correct_bias']:
    framework.compute_correction_weights()

framework.visualize_weights()
exit()
# Framework with CCP ##
# framework.ccp_correct()

## Framework with classic semi-supervised ##
framework.classic_correct()

uncorr_roc = framework.evaluate_uncorrected()
corr_roc = framework.evaluate_corrected()
uncorrected_rocs_y.append(uncorr_roc)
corrected_rocs_y.append(corr_roc)

print("Final mean test ROC AUC (uncorrected): {}".format(np.array(uncorrected_rocs_y).mean()))
print("Final mean test ROC AUC (corrected): {}".format(np.array(corrected_rocs_y).mean()))

print("Final mean S ROC : {}".format(np.array(rocs_s).mean()))
print("Final mean S brier: {}".format(np.array(briers_s).mean()))
