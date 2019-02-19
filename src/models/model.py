# Imports
import pandas as pd
import numpy as np
import math

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, average_precision_score, recall_score

# from yellowbrick.features import RFECV
from sklearn.feature_selection import RFECV

from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Venn ABERS: As implemented by Paolo Toccaceli (https://github.com/ptocca/VennABERS.git)
from ../venn-abers/VennABERS import ScoresToMultiProbs


def import_data(format='numpy_array'):
    data = pd.read_csv(
        '/Users/steliosrammos/Documents/Education/Maastricht/DKE-Year3/Bachelor Thesis/data/v2/data_merged_2018.csv',
        sep=';')
    data = data.fillna(data.mean().apply(lambda x: math.floor(x)))
    # data.head()

    X, y, s = None, None, None

    X = data.iloc[:, 1:-2]
    s = data.iloc[:, -2]
    y = data.iloc[:, -1]

    # Nunmpy Arrays
    if format == 'numpy_array':
        X = X.values
        s = s.values
        y = y.values

    return X, s, y


def get_best_rfc_estimator(X_train, X_test, s_train, s_test, verbose=False):
    skf_2 = StratifiedKFold(2, random_state=1)
    skf_10 = StratifiedKFold(10, shuffle=True, random_state=1)

    clf = RFECV(RandomForestClassifier(n_estimators=100, random_state=1), step=5, cv=skf_2, scoring="roc_auc")
    clf.fit(X_train, s_train)
    mask = clf.support_

    X_train = X_train[:, mask]
    X_test = X_test[:, mask]
    # Parameters for the grid search
    parameters = {
        'random_state': [1],
        'n_jobs': [-1],
        'n_estimators': [150, 300],
        'criterion': ['entropy', 'gini'],
        'bootstrap': [True],
    }

    rfc = RandomForestClassifier()
    grid = GridSearchCV(rfc, param_grid=parameters, cv=skf_10, scoring="roc_auc", verbose=0)
    grid.fit(X_train, s_train)

    if verbose:
        predicted_s = grid.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(s_test, predicted_s)

        print(
            'Best parameters: {}'.format(grid.best_params_),
            'ROC {}'.format(roc_auc),
            'Accuracy {}'.format(accuracy_score(s_test, predicted_s)),
            'Precision {}'.format(precision_score(s_test, predicted_s)),
            'Average precision-recall score: {0:0.2f}'.format(average_precision_score(s_test, predicted_s))
        )

    return grid.best_estimator_, grid.best_score_

def get_best_svm_estimator(X_train, X_test, s_train, s_test, verbose=False):
    # skf_2 = StratifiedKFold(2, random_state=1)
    # skf_10 = StratifiedKFold(10, shuffle=True, random_state=1)
    #
    # clf = RFECV(RandomForestClassifier(n_estimators=100, random_state=1), step=5, cv=skf_2, scoring="roc_auc")
    # clf.fit(X_train, s_train)
    # mask = clf.support_
    #
    # X_train = X_train[:, mask]
    # X_test = X_test[:, mask]
    # # Parameters for the grid search
    # parameters = {
    #     'random_state': [1],
    #     'n_jobs': [-1],
    #     'n_estimators': [150, 300],
    #     'criterion': ['entropy', 'gini'],
    #     'bootstrap': [True],
    # }
    #
    # rfc = RandomForestClassifier()
    # grid = GridSearchCV(rfc, param_grid=parameters, cv=skf_10, scoring="roc_auc", verbose=0)
    # grid.fit(X_train, s_train)
    #
    # if verbose:
    #     predicted_s = grid.predict_proba(X_test)[:, 1]
    #     roc_auc = roc_auc_score(s_test, predicted_s)
    #     print(
    #         'Best parameters: {}'.format(grid.best_params_),
    #         'ROC {}'.format(roc_auc),
    #         'Accuracy {}'.format(accuracy_score(s_test, predicted_s)),
    #         'Precision {}'.format(precision_score(s_test, predicted_s)),
    #         'Average precision-recall score: {0:0.2f}'.format(average_precision_score(s_test, predicted_s))
    #     )

    # return grid.best_estimator_, grid.best_score_

    return 'Not implemented yet'


def classify_rfc(X_train, X_test, s_train, s_test, rfc=None):
    # Parameters for the grid search
    if rfc is None:
        rfc = RandomForestClassifier(bootstrap=True, criterion="gini", n_estimators=100, n_jobs=-1, random_state=1,
                                 oob_score=True)

    estimator = rfc.fit(X_train, s_train)
    predicted_proba = estimator.predict_proba(X_test)
    predicted_s = predicted_proba[:, 1]
    training_scores = estimator.oob_decision_function_

    print(
        'ROC {}'.format(roc_auc_score(s_test, predicted_s))
    )

    return predicted_proba, training_scores


def calibrate(test_scores, train_scores):
    calibr_pts = []

    for i in range(0, train_scores.shape[0]):
        label = np.argmax(train_scores[i, :])
        score = train_scores[i, label]
        calibr_pts.append([score, label])

    p0, p1 = ScoresToMultiProbs(calibr_pts, test_scores)
    print(p0, p1)
    return p0, p1


def build_model(X, y, classifier):
    # Select the correct classifier
    if classifier == 'RandomForestClassifier':
        classify_function = get_best_rfc_estimator
    elif classifier == 'SVM':
        classify_function = get_best_svm_estimator
    else:
        return 'The classifier provided is not handled'

    skf = StratifiedKFold(10, shuffle=True, random_state=1)
    roc_scores = []
    estimators = []
    i = 0
    predicted_proba_trial = None

    # Debug variables
    trial = True
    trial_iterations = 1

    # Cross validation training
    for train_index, test_index in skf.split(X, y):
        if i == trial_iterations - 1:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if trial:
                test_score, training_score = classify_rfc(X_train, X_test, y_train, y_test)
                print('Training set scores shape: {} \n'.format(training_score.shape))
                print('Predicted proba shape: {} \n'.format(test_score.shape))
                calibrate(test_score, training_score)
            else:
                estimator, roc_score = classify_function(X_train, X_test, y_train, y_test)
                print('ROC AUC in CV {}: {} \n'.format(i, roc_score))

            i += 1

        estimators.append(estimator)
        roc_scores.append(roc_score)

    return estimators, roc_scores

def main():
    X, s, _ = import_data('numpy_array')
    build_model(X, s, 'RandomForestClassifier')