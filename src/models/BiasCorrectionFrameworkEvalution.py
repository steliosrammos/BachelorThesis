from ConformalBiasCorrection import ConformalBiasCorrection

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold

# Classifier S
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import time

# Visualizations
import sys
sys.path.append('/Users/steliosrammos/Documents/Education/Maastricht/DKE-Year3/BachelorThesis/bachelor_thesis/src/visualization')
import visualize

import warnings
warnings.filterwarnings('ignore')

# Import data
base_path = '/Users/steliosrammos/Documents/Education/Maastricht/DKE-Year3/BachelorThesis/bachelor_thesis/'

# Select dataset
data = pd.read_csv(base_path+"data/interim/data_extra_nogo_2018_v2.csv", sep=";")
# print(data.info())
# exit()

# data = pd.read_csv(base_path + "data/interim/data_2018 (with bdi-csi).csv", sep=";")
# data = pd.read_csv(base_path+'data/interim/data_extra_nogo_2018_with_missing.csv', sep=";")

# Add weight column with default 1
data.insert(data.shape[1] - 2, "weight", pd.Series())
data.weight = 1

counts = data.got_go.value_counts()
ratio = counts[0]/counts[1]

######### CLASSIFIER S #############
# XGBoost
classifier_s = XGBClassifier()
parameters_s = {'n_jobs': -1, 'reg_lambda': 0.8, 'max_delta_step': 2, 'learing_rate': 0.3, 'max_depth': 6, 'n_estimators': 50, 'objective':'binary:logistic', 'random_state': 55}

# Random Forest
# classifier_s = RandomForestClassifier()
# # parameters_s = {'class_weight': 'balanced', 'n_estimators': 100, 'max_depth': 2, 'random_state': 55}
# parameters_s = {'n_estimators': 100, 'max_depth': 2, 'random_state': 55}

# SVC
# classifier_s = SVC()
# parameters_s = {'gamma': 'auto'}

# Decision Tree
# classifier_s = DecisionTreeClassifier()
# parameters_s = {'random_state': 55}

######### CLASSIFIER y #############
# XGBoost
classifier_y = XGBClassifier()
parameters_y = {'n_jobs': -1, 'learning_rate': 0.3, 'max_depth': 3, 'n_estimators': 10, 'scale_pos_weight': 0.5, 'objective':'binary:logistic', 'random_state': 55}

# Random Forest
# classifier_s = RandomForestClassifier()
# # parameters_s = {'class_weight': 'balanced', 'n_estimators': 100, 'max_depth': 2, 'random_state': 55}
# parameters_s = {'n_estimators': 100, 'max_depth': 2, 'random_state': 55}

# SVC
# classifier_s = SVC()
# parameters_s = {'gamma': 'auto'}

# Decision Tree
# classifier_s = DecisionTreeClassifier()
# parameters_s = {'random_state': 55}


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
    # 'rebalancing_s': 'SMOTE',
    # 'rebalancing_s': 'oversampling',
    # 'rebalancing_s': 'undersampling',
    'SMOTE_y': False,
    'conformal_oversampling': True,
    'feature_selection': True}

bias_correction_parameters = {'correct_bias': True}

uncorrected_roc = []
corrected_roc = []

start = time.time()
num_runs = 1

for i in range(0, num_runs):

    # Initialize some variables
    sss = StratifiedKFold(n_splits=10)

    X = data.iloc[:, :-2]
    y = data.iloc[:, -2]

    # all_feature_importances = []
    uncorrected_rocs_y = []
    corrected_rocs_y = []

    rocs_s = []
    briers_s = []

    for train_index, valid_index in sss.split(X, y):
        data_train = data.loc[train_index]
        data_test = data.loc[valid_index]

        framework = ConformalBiasCorrection(train_data=data_train, test_data=data_test, classifiers=classifiers, clf_parameters=clf_parameters, rebalancing_parameters=rebalancing_parameters, bias_correction_parameters=bias_correction_parameters)
        framework.verbose = 1
        framework.random_state = 55

        if bias_correction_parameters['correct_bias']:
            roc, brier = framework.compute_correction_weights()
            rocs_s.append(roc)
            briers_s.append(brier)

        # Framework with CCP ##
        # framework.ccp_correct()

        ## Framework with classic semi-supervised ##
        # framework.classic_correct()

        uncorr_roc, corr_roc = framework.final_evaluation()
        uncorrected_rocs_y.append(uncorr_roc)
        corrected_rocs_y.append(corr_roc)
        # all_feature_importances.append(feature_importances)

    uncorrected_roc.append(np.array(uncorrected_rocs_y).mean())
    corrected_roc.append(np.array(corrected_rocs_y).mean())

    print("Final mean test ROC AUC (uncorrected): {}".format(np.array(uncorrected_rocs_y).mean()))
    print("Final mean test ROC AUC (corrected): {}".format(np.array(corrected_rocs_y).mean()))

    print("Final mean S ROC : {}".format(np.array(rocs_s).mean()))
    print("Final mean S brier: {}".format(np.array(briers_s).mean()))

    # Visualize
    # importance_ratios = np.array(all_feature_importances).mean(axis=0)
    # visualize.important_features(importance_ratios)
    # print(importance_ratios)

# end = time.time()
# total = end-start
# print("Completed after {} seconds.".format(total))
#
# t_score, p_value = stats.ttest_ind(uncorrected_roc, corrected_roc, equal_var=False)
# print("Mean uncorrected ROC: {}".format(np.array(uncorrected_roc).mean()))
# print("Mean corrected ROC: {}".format(np.array(corrected_roc).mean()))
# print("T-score: {} ".format(t_score))
# print("P-value: {} \n".format(p_value))
#
# print(uncorrected_roc)
# print(corrected_roc)
