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
data = pd.read_csv(base_path+"data/interim/data_extra_nogo_2018_v2.csv", sep=";")

# data = pd.read_csv(base_path + "data/interim/data_2018 (with bdi-csi).csv", sep=";")
data.insert(data.shape[1] - 2, "weight", pd.Series())
# data.iloc[:, :-3] = data.iloc[:, :-3].fillna(data.iloc[:, :-3].mean())
data.weight = 1
# print(data.info())
# # exit()
# print(data["got_go"].value_counts())
# exit()

# Set classifiers and their parameters for the framework

classifier_s = XGBClassifier()

# parameters_s = {'n_jobs': 6, 'reg_lambda': 0.8, 'scale_pos_weight': 0.2, 'learing_rate': 0.05, 'max_depth': 30, 'n_estimators': 150,'objective':'binary:logistic'} --> BL: 0.2053
# parameters_s = {'n_jobs': 6, 'reg_lambda': 0.8, 'scale_pos_weight': 0.3487, 'learing_rate': 0.3, 'max_depth': 6, 'n_estimators': 50, 'objective':'binary:logistic'} # --> BL: 0.2025
counts = data.got_go.value_counts()
ratio = counts[0]/counts[1]
parameters_s = {'n_jobs': 6, 'reg_lambda': 0.8, 'scale_pos_weight': ratio, 'learing_rate': 0.3, 'max_depth': 6, 'n_estimators': 50, 'objective':'binary:logistic'} # --> BL: 0.2025

# classifier_s = RandomForestClassifier()
# parameters_s = {'n_jobs': 6, 'max_depth': 30, 'n_estimators': 150, 'random_state': 55}

# classifier_s = SVC()
# parameters_s = {'probability':True, 'random_state':55}

classifier_y = XGBClassifier()
# parameters_y = {'n_jobs': 6, 'learning_rate': 0.3, 'max_depth': 3, 'n_estimators': 10, 'scale_pos_weight': 0.2, 'objective':'binary:logistic'} // ROC: 0.5848 -> 0.6253 with factor 2 // ROC: 0.5848 -> 0.6390 with factor 3 // ROC: 0.5848 -> 0.6388 with factor 4 // ROC : 0.5848 -> 0.6497 with 4 and 0.6
parameters_y = {'n_jobs': 6, 'learning_rate': 0.3, 'max_depth': 3, 'n_estimators': 10, 'scale_pos_weight': 0.2, 'objective':'binary:logistic', 'random_state':55}

classifiers = {
    'classifier_s': classifier_s,
    'classifier_y': classifier_y
}

clf_parameters = {
    'classifier_s': parameters_s,
    'classifier_y': parameters_y
}

rebalancing_parameters = {'SMOTE_s': True, 'SMOTE_y': False, 'conformal_oversampling': True}

# Run the framework
sss = StratifiedKFold(n_splits=5, random_state=1)

X = data.iloc[:, :-2]
y = data.iloc[:, -2]

uncorrected_rocs = []
corrected_rocs = []

# framework = ConformalBiasCorrection(train_data=data, test_data=[], classifiers=classifiers, clf_parameters=clf_parameters, rebalancing_parameters=rebalancing_parameters, verbose=3)
# framework.compute_correction_weights()
#
for train_index, valid_index in sss.split(X, y):
    data_train = data.loc[train_index]
    data_test = data.loc[valid_index]

    framework = ConformalBiasCorrection(train_data=data_train, test_data=data_test, classifiers=classifiers, clf_parameters=clf_parameters, rebalancing_parameters=rebalancing_parameters, verbose=3)
    framework.compute_correction_weights()
    framework.ccp_correct()

    uncorrected_roc, corrected_roc = framework.final_evaluation()
    uncorrected_rocs.append(uncorrected_roc)
    corrected_rocs.append(corrected_roc)

print("Final mean test ROC AUC (uncorrected): {}".format(np.array(uncorrected_rocs).mean()))
print("Final mean test ROC AUC (corrected): {}".format(np.array(corrected_rocs).mean()))

# exit()

