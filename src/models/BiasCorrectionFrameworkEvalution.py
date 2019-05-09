from ConformalBiasCorrection import ConformalBiasCorrection

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

# Import data
base_path = '/Users/steliosrammos/Documents/Education/Maastricht/DKE-Year3/BachelorThesis/bachelor_thesis/'

# Select dataset
# data = pd.read_csv(base_path+"data/interim/data_extra_nogo_2018_v2.csv", sep=";")
# data = pd.read_csv(base_path + "data/interim/data_2018 (with bdi-csi).csv", sep=";")
data = pd.read_csv(base_path+'data/interim/data_extra_nogo_2018_with_missing.csv', sep=";")

# Add weight column with default 1
data.insert(data.shape[1] - 2, "weight", pd.Series())
data.weight = 1

# Set classifiers and their parameters for the framework


# parameters_s = {'n_jobs': 6, 'reg_lambda': 0.8, 'scale_pos_weight': 0.2, 'learing_rate': 0.05, 'max_depth': 30, 'n_estimators': 150,'objective':'binary:logistic'} --> BL: 0.2053
# parameters_s = {'n_jobs': 6, 'reg_lambda': 0.8, 'scale_pos_weight': 0.3487, 'learing_rate': 0.3, 'max_depth': 6, 'n_estimators': 50, 'objective':'binary:logistic'} # --> BL: 0.2025
counts = data.got_go.value_counts()
ratio = counts[0]/counts[1]

######### CLASSIFIER S #############
# XGBoost
classifier_s = XGBClassifier()
parameters_s = {'n_jobs': -1, 'reg_lambda': 0.8, 'max_delta_step': 2, 'learing_rate': 0.3, 'max_depth': 6, 'n_estimators': 50, 'objective':'binary:logistic', 'random_state': 55}

######### CLASSIFIER y #############
# XGBoost
classifier_y = XGBClassifier()
parameters_y = {'n_jobs': -1, 'learning_rate': 0.3, 'max_depth': 3, 'n_estimators': 10, 'scale_pos_weight': 0.5, 'objective':'binary:logistic', 'random_state': 55}

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
rebalancing_parameters = {'SMOTE_s': False, 'SMOTE_y': False, 'conformal_oversampling': True}
bias_correction_parameters = {'correct_bias': True}

# Initialize some variables
sss = StratifiedKFold(n_splits=10, random_state=1)

X = data.iloc[:, :-2]
y = data.iloc[:, -2]

uncorrected_rocs_y = []
corrected_rocs_y = []

rocs_s = []
briers_s = []

all_feature_importances = []
# framework = ConformalBiasCorrection(train_data=data, test_data=[], classifiers=classifiers, clf_parameters=clf_parameters, rebalancing_parameters=rebalancing_parameters, verbose=3)
# framework.compute_correction_weights()
#
for train_index, valid_index in sss.split(X, y):
    data_train = data.loc[train_index]
    data_test = data.loc[valid_index]

    framework = ConformalBiasCorrection(train_data=data_train, test_data=data_test, classifiers=classifiers, clf_parameters=clf_parameters, rebalancing_parameters=rebalancing_parameters, bias_correction_parameters=bias_correction_parameters, verbose=3)

    if bias_correction_parameters['correct_bias']:
        roc, brier = framework.compute_correction_weights()
        rocs_s.append(roc)
        briers_s.append(brier)
    # framework.ccp_correct()
    #
    # uncorrected_roc, corrected_roc, feature_importances = framework.final_evaluation()
    # uncorrected_rocs_y.append(uncorrected_roc)
    # corrected_rocs_y.append(corrected_roc)
    # all_feature_importances.append(feature_importances)
#
# print("Final mean test ROC AUC (uncorrected): {}".format(np.array(uncorrected_rocs_y).mean()))
# print("Final mean test ROC AUC (corrected): {}".format(np.array(corrected_rocs_y).mean()))
# print(np.array(all_feature_importances))

print("Final mean S ROC : {}".format(np.array(rocs_s).mean()))
print("Final mean S brier: {}".format(np.array(briers_s).mean()))

# exit()

