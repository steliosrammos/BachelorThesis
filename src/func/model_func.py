import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
# from yellowbrick.features import RFECV as RFECViz

from sklearn.ensemble import RandomForestClassifier

from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from src.models.VennABERS import ScoresToMultiProbs

from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, average_precision_score, recall_score

from sklearn.externals import joblib


def save_model(classifier, filename):
    # try:
    filename = filename+'.sav'
    joblib.dump(classifier, filename)

    #     print('Failed to save the classifier!')
    #     return False
    #
    # return True


def load_model(filename):
    model = joblib.load(filename)
    return model


def get_metrics(predicted_s, proba_predicted_s, y_test):
    accuracy = accuracy_score(y_test, predicted_s)
    precision = precision_score(y_test, predicted_s)
    recall = recall_score(y_test, predicted_s)
    roc_score = roc_auc_score(y_test, proba_predicted_s)

    print(
        'Accuracy: {} \n'.format(accuracy),
        'Precision: {} \n'.format(precision),
        'Recall: {} \n'.format(recall),
        'ROC AUC: {} \n'.format(roc_score)
    )


def compute_corr_coeff(prob_s, prob_predicted_s):
    corr_coeff = np.zeros(prob_predicted_s.shape[0])

    for i in range(0, prob_predicted_s.shape[0]):
        corr_coeff[i] = prob_s / prob_predicted_s[i]

    return corr_coeff


def get_best_rfc_estimator(X_train, X_test, y_train, y_test, verbose=False):
    '''
    :param X_train: dataframe or array with training instances
    :param X_test: dataframe or array with test instances
    :param y_train: dataframe or array with training labels
    :param y_test: dataframe or array with testing labels
    :param verbose: set to 'True' for a detailed report of the grid search
    :return: the best estimator and its corresponding score
    '''
    skf_10 = StratifiedKFold(10, shuffle=True, random_state=1)

    eliminate_features(X_train, X_test, y_train)
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
    grid.fit(X_train, y_train)

    if verbose:
        predicted_y = grid.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, predicted_y)

        print(
            'Best parameters: {}'.format(grid.best_params_),
            'ROC {}'.format(roc_auc),
            'Accuracy {}'.format(accuracy_score(y_test, predicted_y)),
            'Precision {}'.format(precision_score(y_test, predicted_y)),
            'Average precision-recall score: {0:0.2f}'.format(average_precision_score(y_test, predicted_y))
        )

    return grid.best_estimator_, grid.best_score_


def eliminate_features(X_train, X_test, y_train):
    # Select best features with Recursive Feature Elimination
    skf_2 = StratifiedKFold(2, random_state=1)
    clf = RFECV(RandomForestClassifier(n_estimators=100, random_state=1), step=5, cv=skf_2, scoring="roc_auc")
    clf.fit(X_train, y_train)
    mask = clf.support_

    X_train = X_train[:, mask]
    X_test = X_test[:, mask]

    return X_train, X_test

def calibrate(test_scores, train_scores):
    calibr_pts = []

    for i in range(0, train_scores.shape[0]):
        label = np.argmax(train_scores[i, :])
        score = train_scores[i, label]
        calibr_pts.append([score, label])

    p0, p1 = ScoresToMultiProbs(calibr_pts, test_scores)
    #     print('p0: {} \n'.format(p0.shape), p0)
    #     print('p1: {} \n'.format(p1.shape), p1)
    #     print(p0,p1)
    p = log_loss(p0, p1)
    #     print('Average probability of predicted s (calibrated): {}'.format(np.mean(p, axis =0)))
    return p


def log_loss(p0, p1):
    p = []

    for i in range(0, p0.shape[0]):
        p0_single = p1[i, 0] / (1 - p0[i, 0] + p1[i, 0])
        p1_single = p1[i, 1] / (1 - p0[i, 1] + p1[i, 1])

        p.append([p0_single, p1_single])

    p = np.array(p)

    return p


def plot_calibration_curve(y_test, prob_y, name, ax):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_y)

    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=name)


def build_model(X, y, classifier, verbose = False):
    # Select the correct classifier
    if classifier == 'RandomForestClassifier':
        classify_function = get_best_rfc_estimator
    # elif classifier == 'SVM':
    #     classify_function = get_best_svm_estimator
    else:
        return 'The classifier provided is not handled'

    # Initiate some variables and parameters
    skf = StratifiedKFold(10, shuffle=True, random_state=1)
    roc_scores = []
    classifiers = []
    i = 0
    predicted_proba_trial = None

    if verbose:
        # Initiate figure for plotting calibration curve
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel("Fraction of positives")
        ax.set_ylim([-0.05, 1.05])
        ax.set_title('Calibration plots  (reliability curve)')
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # Debug variables
    trial = False
    trial_iterations = 10

    # Cross validation training
    for train_index, test_index in skf.split(X, y):
        if i <= trial_iterations - 1:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if trial:
                test_score, training_score, proba_y = classify_rfc(X_train, X_test, y_train, y_test)
                _, calibrated_proba_y = classify_rfc_calibrated(X_train, X_test, y_train, y_test)

                p = calibrate(test_score, training_score)
                custom_calibrated_proba_y = p[:, 1]

                if verbose:
                    print(
                        'Probability of s (test sample): {} \n'.format(np.count_nonzero(y_test)/len(y_test)),
                        'Probability of s (full dataset): {} \n'.format(np.count_nonzero(y)/len(y)),
                        'Training set scores shape: {} \n'.format(training_score.shape),
                        'Predicted proba shape: {} \n'.format(test_score.shape),
                        'ROC calibrated custom: {}'.format(roc_auc_score(y_test, custom_calibrated_proba_y))
                    )

                plot_calibration_curve(y_test, proba_y, 'RFC CV {}'.format(i), ax)
                plot_calibration_curve(y_test, calibrated_proba_y, 'RFC calibrated CV {}'.format(i), ax)
                plot_calibration_curve(y_test, custom_calibrated_proba_y, 'RFC calibrated Venn ABERS CV {}'.format(i),
                                       ax)

            else:
                estimator, roc_score_grid = classify_function(X_train, X_test, y_train, y_test)

                # Calibrate estimator
                isotonic = CalibratedClassifierCV(estimator, cv=2, method='isotonic')
                clf = isotonic.fit(X_train, y_train)

                predicted_proba = isotonic.predict_proba(X_test)
                proba_y = predicted_proba[:, 1]
                roc_score = roc_auc_score(y_test, proba_y)

                if verbose:
                    print(
                        'Grid Search ROC AUC in CV {}: {} \n'.format(i, roc_score_grid),
                        'Calibrated ROC AUC in CV {}: {} \n'.format(i, roc_score),
                        '\n'
                    )
                    # Plot calibration curve
                    plot_calibration_curve(y_test, proba_y, 'RFC calibrated CV {} - ROC {}'.format(i, roc_score), ax)

                classifiers.append(clf)
                roc_scores.append(roc_score)

            i += 1

    if verbose:
        plt.legend(loc="best")
        plt.show()

    if not trial:
        best_index = np.argmax(roc_scores)
        best_roc_score = roc_scores[best_index]
        best_classifier = classifiers[best_index]

        return best_classifier, best_roc_score

    return None, None

#########################################################################

# THE FUNCTIONS BELOW ARE ONLY USED FOR EXPLORATION AND/OR TESTING PHASES

#########################################################################


def classify_rfc(X_train, X_test, y_train, y_test, rfc=None):
    '''
    This function classifies the training instances with an uncalibrated classifier
    :param X_train: dataframe or array with training instances
    :param X_test: dataframe or array with test instances
    :param y_train: dataframe or array with training labels
    :param y_test: dataframe or array with testing labels
    :param rfc: random forest classifier, default=None
    :return: the predicted class probabilities, the estimator scores and the predicted y probabilities
    '''
    # Parameters for the grid search
    if rfc is None:
        rfc = RandomForestClassifier(bootstrap=True, criterion="gini", n_estimators=100, n_jobs=-1, random_state=1,
                                     oob_score=True)

    estimator = rfc.fit(X_train, y_train)
    predicted_proba = estimator.predict_proba(X_test)

    proba_y = predicted_proba[:, 1]
    training_scores = estimator.oob_decision_function_

    print(
        'Average probability of predicted s (non-calibrated): {}'.format(np.mean(proba_y, axis=0)),
        'ROC uncalibrated: {}'.format(roc_auc_score(y_test, proba_y))
    )

    return predicted_proba, training_scores, proba_y


def classify_rfc_calibrated(X_train, X_test, y_train, y_test, rfc=None):
    '''
    This function classifies the training instances with a calibrated classifier
    :param X_train: dataframe or array with training instances
    :param X_test: dataframe or array with test instances
    :param y_train: dataframe or array with training labels
    :param y_test: dataframe or array with testing labels
    :param rfc: random forest classifier, default=None
    :return: the predicted class probabilities, the predicted y probabilities
    '''
    # Parameters for the grid search
    if rfc is None:
        rfc = RandomForestClassifier(bootstrap=True, criterion="gini", n_estimators=100, n_jobs=-1, random_state=1,
                                     oob_score=True)

    #     estimator = rfc.fit(X_train, y_train)
    isotonic = CalibratedClassifierCV(rfc, cv=2, method='isotonic')
    estimator = isotonic.fit(X_train, y_train)

    predicted_proba = isotonic.predict_proba(X_test)

    proba_y = predicted_proba[:, 1]

    #     print(
    #         'Average probability of predicted s (non-calibrated): {} \n'.format(np.mean(proba_y, axis=0)),
    #         'ROC calibrated: {}'.format(roc_auc_score(y_test, proba_y))
    #     )

    return predicted_proba, proba_y