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
data = pd.read_csv(base_path+'data/external/parkinson.csv', sep=';')

got_go = pd.Series(np.ones(data.shape[0]))
data.insert(data.shape[1] - 1, value=got_go, column='got_go')

# print(data.shape[0])
# exit()

data['weight'] = 1

X = data.drop(['class', 'got_go', 'weight'], axis=1)
y = data.loc[:, 'class']

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
    'balance_new_labels': False
}

bias_correction_parameters = {
    'correct_bias': True,
    'dynamic_weights': False
}

uncorrected_rocs_y = []
corrected_rocs_y = []

rocs_s = []
briers_s = []
num_runs = 1

all_labeled = []

skf = StratifiedKFold(30, random_state=None)

for train_index, test_index in skf.split(X, y):

    train_data = data.iloc[train_index, :]
    test_data = data.iloc[test_index, :]

    biased_train_data = train_data.copy()
    biased_train_data.loc[(biased_train_data['MDVP:Fo(Hz)'] < 120) | (biased_train_data['MDVP:Flo(Hz)'] > 130), 'class'] = np.nan
    biased_train_data.loc[(biased_train_data['MDVP:Fo(Hz)'] < 120) | (biased_train_data['MDVP:Flo(Hz)'] > 130), 'got_go'] = 0

    framework = ConformalBiasCorrection(train_data=biased_train_data, test_data=test_data, classifiers=classifiers,
                                        clf_parameters=clf_parameters,
                                        rebalancing_parameters=rebalancing_parameters,
                                        bias_correction_parameters=bias_correction_parameters)
    framework.verbose = 1
    framework.random_state = None

    if bias_correction_parameters['correct_bias']:
        framework.compute_correction_weights()

    ## Framework with classic semi-supervised ##
    # framework.classic_correct(0.8)

    ## Framework with CCP ##
    framework.ccp_correct(0.8)

    uncorr_roc = framework.evaluate_uncorrected()
    corr_roc = framework.evaluate_corrected()

    uncorrected_rocs_y.append(uncorr_roc)
    corrected_rocs_y.append(corr_roc)

t_score, p_value = stats.ttest_ind(uncorrected_rocs_y, corrected_rocs_y, equal_var=False)
print('Mean uncorrected ROC: {}'.format(np.array(uncorrected_rocs_y).mean()))
print('Mean corrected ROC: {}'.format(np.array(corrected_rocs_y).mean()))
print('T-score: {} '.format(t_score))
print('P-value: {} \n'.format(p_value))