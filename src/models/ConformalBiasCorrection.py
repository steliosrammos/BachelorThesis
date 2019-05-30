import pandas as pd
import numpy as np
import random
from imblearn.over_sampling import SMOTE
from nonconformist.icp import IcpClassifier
from nonconformist.nc import NcFactory, MarginErrFunc
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, accuracy_score, brier_score_loss, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import matplotlib.pyplot as plt

class ConformalBiasCorrection:

    def __init__(self, train_data, test_data, classifiers, clf_parameters, rebalancing_parameters, bias_correction_parameters, verbose=0, random_state=None):

        self.train_data = train_data
        self.test_data = test_data

        # Defined after self training
        self.augmented_data_lbld = None

        self.classifiers = classifiers
        self.clf_parameters = clf_parameters
        self.rebalancing_parameters = rebalancing_parameters
        self.bias_correction_parameters = bias_correction_parameters
        self.verbose = verbose
        self.random_state = random_state

    def compute_correction_weights(self):

        print('Start computing weights W')

        data_s = self.train_data.drop('class', axis=1)

        # Shuffle rows
        X = data_s.iloc[:, :-2]
        y = data_s.iloc[:, -1]

        # Startified k-fold
        sss = StratifiedKFold(n_splits=10, random_state=1)

        if type(y) == np.ndarray:
            prob_s_pos = np.bincount(y)[1] / len(y)
        else:
            prob_s_pos = y.value_counts(True)[1]

        predicted_prob_s = pd.Series(np.full(y.shape[0], np.nan))
        predicted_prob_s.index = y.index

        fold_num = 0

        for train_index, valid_index in sss.split(X, y):

            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            # Compute weights for test examples
            clf_s = self.classifiers["classifier_s"]
            clf_s.fit(X_train, y_train)

            predicted_prob_s.loc[X_valid.index] = clf_s.predict_proba(X_valid)[:, 1]

            for index in X_valid.index:
                if predicted_prob_s.loc[index] > 0:
                    self.train_data.loc[index, 'weight'] = prob_s_pos / predicted_prob_s.loc[index]
                else:
                    self.train_data.loc[index, 'weight'] = 0.001 / prob_s_pos
            fold_num += 1

            data_y = self.train_data.drop('got_go', axis=1)
            data_lbld = data_y[~data_y["class"].isna()]
            self.augmented_data_lbld = data_lbld

        return True

    def visualize_weights(self):
        data = self.augmented_data_lbld
        weights = data.weight

        plt.hist(weights, bins=100)
        plt.show()

    # Classical semi-supervised learning
    def classic_correct(self):

        data_y = self.train_data.drop(['got_go'], axis=1)

        # Split data into labeled and unlabeled sets
        data_lbld = data_y[~data_y["class"].isna()]

        data_unlbld = data_y[data_y["class"].isna()]
        total_unlbld = data_unlbld.shape[0]

        predictions = self.classic_predict(data_lbld, data_unlbld)

        data_unlbld["class"] = predictions

        num_new_labels_to_select = int(data_unlbld.shape[0]*0.1)
        self.augmented_data_lbld = data_lbld.append(data_unlbld.sample(num_new_labels_to_select))
        self.augmented_data_lbld

    def classic_predict(self, data_lbld, data_unlbld):
        # Create SMOTE instance for class rebalancing
        smote = SMOTE(random_state=self.random_state)

        # Create instance of classifier
        classifier_y = self.classifiers['classifier_y']
        parameters_y = self.clf_parameters['classifier_y']

        clf = classifier_y.set_params(**parameters_y)

        X = data_lbld.iloc[:, :-2]
        y = data_lbld.iloc[:, -1]

        X_unlbld = data_unlbld.iloc[:, :-2]

        if self.rebalancing_parameters['SMOTE_y']:
            X, y = smote.fit_resample(X, y)
            clf.fit(X[:, :-1], y, sample_weight=X[:, -1])
        else:
            clf.fit(X.iloc[:, :-1], y, sample_weight=X.iloc[:, -1])

        predictions = clf.predict(X_unlbld.iloc[:, :-1])

        return predictions

    # CROSS-VALIDATED CONFORMAL PREDICTIONS
    def ccp_correct(self):

        data_y = self.train_data.drop(['got_go'], axis=1)

        # Split data into labeled and unlabeled sets
        data_lbld = data_y[~data_y["class"].isna()]

        data_unlbld = data_y[data_y["class"].isna()]
        total_unlbld = data_unlbld.shape[0]

        # Compute positive to negative ratio
        label_cnts = data_lbld["class"].value_counts()
        ratio = label_cnts[0] / label_cnts[1]

        if self.rebalancing_parameters['conformal_oversampling']:
            over_sampling_factor = int(label_cnts[1] / label_cnts[0])

        # Initialize array of newly_labeled data
        new_lbld = data_unlbld.copy()
        all_newly_labeled_indeces = []
        last_newly_labeled_indeces = []

        # Initialize stopping variable
        stop = False

        if self.verbose >= 2:
            print('Start conformal improvement.')
            print("Initial unlabeled: {} \n".format(data_unlbld.shape[0]))
            print("Initial ratio: {} \n".format(ratio))

        iterations = 0

        while not stop:

            # Make conformal predictions
            ccp_predictions = self.ccp_predict(data_lbld, data_unlbld, new_lbld.loc[all_newly_labeled_indeces])

            # Add best predictions
            labels = self.get_best_pred_indeces(ccp_predictions, 0.6, ratio)
            new_lbld.loc[labels.index.values, 'class'] = labels.values

            # Save new label's indeces
            newly_labeled_indeces = list(labels.index.values)

            if self.rebalancing_parameters['conformal_oversampling']:
                if ratio <= 0.5:
                    oversampled_newly_lbld_indeces = self.oversample_minority(new_lbld, 'class', 0, over_sampling_factor, newly_labeled_indeces)
                    all_newly_labeled_indeces += oversampled_newly_lbld_indeces
            else:
                all_newly_labeled_indeces += newly_labeled_indeces

            if self.verbose >= 2:
                print('Number of good predictions: {} \n'.format(labels.shape[0]))

            remain_unlbld = data_unlbld.shape[0]

            if data_unlbld.shape[0] > 0 and labels.shape[0] != 0 and remain_unlbld > total_unlbld * 0.8:
                iterations += 1
                last_newly_labeled_indeces = all_newly_labeled_indeces

                data_unlbld = data_unlbld.drop(newly_labeled_indeces)

                # Update ratio
                new_ratio = self.calculate_ratio(data_lbld, new_lbld)
                ratio = new_ratio

                if self.verbose >= 2:
                    print("Updated ratio: {} \n".format(ratio))
                    print("Remaining unlabeled: {}".format(data_unlbld.shape[0]))

            else:

                if self.verbose >= 2:
                    print("Did not improve...")

                stop = True

        self.augmented_data_lbld = data_lbld.append(new_lbld.loc[last_newly_labeled_indeces])

    def ccp_predict(self, data_lbld, data_unlbld, new_lbld):

        # Create SMOTE instance for class rebalancing
        smote = SMOTE(random_state=self.random_state)

        # Create instance of classifier
        classifier_y = self.classifiers['classifier_y']
        parameters_y = self.clf_parameters['classifier_y']

        clf = classifier_y.set_params(**parameters_y)

        X = data_lbld.iloc[:, :-2]
        y = data_lbld.iloc[:, -1]
        w = data_lbld.iloc[:, -2]

        X_new = new_lbld.iloc[:, :-2]
        y_new = new_lbld.iloc[:, -1]
        w_new = new_lbld.iloc[:, -2]

        X = X.append(X_new, sort=False)
        y = y.append(y_new)
        w = w.append(w_new)

        X_unlbld = data_unlbld.iloc[:, :-2]

        sss = StratifiedKFold(n_splits=5, random_state=self.random_state)
        sss.get_n_splits(X, y)

        p_values = []

        for train_index, calib_index in sss.split(X, y):
            X_train, X_calib = X.iloc[train_index], X.iloc[calib_index]
            y_train, y_calib = y.iloc[train_index], y.iloc[calib_index]
            w_train = w.iloc[train_index]

            if self.rebalancing_parameters['SMOTE_y']:
                X_train, y_train = smote.fit_resample(X_train, y_train)
                clf.fit(X_train, y_train, sample_weight=w_train)
            else:
                clf.fit(X_train.values, y_train, sample_weight=w_train)

            nc = NcFactory.create_nc(clf, MarginErrFunc())
            icp = IcpClassifier(nc)

            if self.rebalancing_parameters['SMOTE_y']:
                icp.fit(X_train, y_train)
            else:
                icp.fit(X_train.values, y_train)

            icp.calibrate(X_calib.values, y_calib)

            # Predict confidences for validation sample and unlabeled sample
            p_values.append(icp.predict(X_unlbld.values, significance=None))

        mean_p_values = np.array(p_values).mean(axis=0)
        ccp_predictions = pd.DataFrame(mean_p_values, columns=['mean_p_0', 'mean_p_1'])
        ccp_predictions["credibility"] = [row.max() for _, row in ccp_predictions.iterrows()]
        ccp_predictions["confidence"] = [1-row.min() for _, row in ccp_predictions.iterrows()] #ccp_predictions.apply(lambda x: 1 - x.max(axis=1), axis=1)
        ccp_predictions["criteria"] = ccp_predictions["credibility"]*ccp_predictions["confidence"]
        ccp_predictions.index = X_unlbld.index

        return ccp_predictions

    # def get_best_estimator(self, classifiers_s, parameters, X_train, X_valid, y_train, y_valid, grid_search=False):
    #
    #     if grid_search:
    #         skf_5 = StratifiedKFold(5, random_state=self.random_state)
    #         estimator = GridSearchCV(classifiers_s, param_grid=parameters, cv=skf_5, scoring="roc_auc", verbose=0)
    #
    #     else:
    #         estimator = classifiers_s.set_params(**parameters)
    #
    #     rebalancing = self.rebalancing_parameters['rebalancing_s']
    #
    #     if rebalancing is not None:
    #         if rebalancing == 'SMOTE':
    #             rebalance = SMOTE(0.9, random_state=self.random_state)
    #         elif rebalancing == 'oversampling':
    #             rebalance = RandomOverSampler()
    #         elif rebalancing == 'undersampling':
    #             rebalance = RandomUnderSampler()
    #
    #         X_train, y_train = rebalance.fit_resample(X_train, y_train)
    #
    #     # Calibrate the classifier here
    #     calib_estimator = CalibratedClassifierCV(estimator, cv=5, method='isotonic')
    #     calib_estimator.fit(X_train, y_train)
    #
    #     predicted_proba = calib_estimator.predict_proba(X_valid)[:, 1]
    #     predicted_labels = calib_estimator.predict(X_valid)
    #     roc_auc = roc_auc_score(y_valid, predicted_proba)
    #     pr_auc = average_precision_score(y_valid, predicted_proba)
    #     brier_loss = brier_score_loss(y_valid, predicted_labels)
    #
    #     if self.verbose >= 4:
    #         print(
    #             'ROC AUC: {0:0.2f} \n'.format(roc_auc),
    #             'PR AUC: {0:0.2f} \n'.format(pr_auc),
    #             'Accuracy {0:0.2f} \n '.format(accuracy_score(y_valid, predicted_labels)),
    #             'Precision {0:0.2f}\n '.format(precision_score(y_valid, predicted_labels)),
    #             'Brier Loss {0:0.2f}\n '.format(brier_score_loss(y_valid, predicted_labels))
    #         )
    #
    #     return calib_estimator, roc_auc, pr_auc, brier_loss

    @staticmethod
    def calculate_ratio(data_lbld, new_lbld):
        counts_data = data_lbld["class"].value_counts()
        counts_new = new_lbld["class"].value_counts()

        negatives = counts_data[0]
        positives = counts_data[1]

        if len(counts_new) == 2:
            negatives = counts_data[0] + counts_new[0]
            positives = counts_data[1] + counts_new[1]

        ratio = negatives / positives

        return ratio

    def  get_best_pred_indeces(self, predictions, threshold, true_ratio):
        '''
            Returns the predictions that have a confidence level above a given threshold.
            The labels returned will have the same positive/negative ratio as the ratio specified by "true_ratio".
            true_ratio: num_negatives / num_positives
        '''

        predictions['labels'] = [np.argmax(row.values[0:2]) for _, row in predictions.iterrows()]

        positives = predictions[predictions["labels"] == 1].sort_values("mean_p_1")
        negatives = predictions[predictions["labels"] == 0].sort_values("mean_p_0")

        positives = positives[positives["criteria"] >= threshold]
        negatives = negatives[negatives["criteria"] >= threshold]

        current_ratio = (negatives.shape[0] + 1) / (positives.shape[0] + 1)

        if self.rebalancing_parameters['balance_new_labels'] == True:
            if current_ratio < true_ratio:
                while current_ratio < true_ratio and positives.shape[0] > 1:
                    positives = positives[1:]
                    current_ratio = (negatives.shape[0] + 1) / (positives.shape[0] + 1)

            elif current_ratio > true_ratio:
                while current_ratio > true_ratio and negatives.shape[0] > 1:
                    negatives = negatives[1:]
                    current_ratio = (negatives.shape[0] + 1) / (positives.shape[0] + 1)

        positives_indeces = list(positives.index.values)
        negatives_indeces = list(negatives.index.values)

        if self.verbose >= 2:
            print("Good positive predictions: {} \n".format(len(positives_indeces)))
            print("Good negative predictions: {} \n".format(len(negatives_indeces)))

        best_pred_indeces = positives_indeces + negatives_indeces
        labels = predictions.loc[best_pred_indeces, "labels"]

        return labels

    def oversample_minority(self, dataframe, label_column, minority_class, factor, at_indeces=None):
        oversampled_indeces = []
        num_oversampled = 0

        if at_indeces is not None:
            for index in at_indeces:
                if dataframe.loc[index, label_column] == minority_class:
                    oversampled_indeces += factor * [index]
                    num_oversampled += 1
                else:
                    oversampled_indeces += [index]
        else:
            for index in dataframe.index:
                if dataframe.loc[index, label_column] == minority_class:
                    oversampled_indeces += factor * [index]
                    num_oversampled += 1
                else:
                    oversampled_indeces += [index]

        if self.verbose >= 2:
            print('Oversampled {} examples by factor of {}.'.format(num_oversampled, factor))

        return oversampled_indeces

    def evaluate_uncorrected(self):

        data_y = self.train_data.drop(['got_go'], axis=1)
        data_lbld = data_y[~data_y["class"].isna()]

        uncorrected_train_data = data_lbld.drop('weight', axis=1)
        test_data = self.test_data

        X_train_uncorrected = uncorrected_train_data.iloc[:, :-1]
        y_train_uncorrected = uncorrected_train_data.iloc[:, -1]

        test_data = test_data.dropna(subset=['class'])
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]

        model = self.classifiers['classifier_y']

        skf_5 = StratifiedKFold(5, random_state=1)
        calibrated_model = CalibratedClassifierCV(model, cv=skf_5, method='isotonic')

        # Uncorrected model evaluation
        calibrated_model.fit(X_train_uncorrected, y_train_uncorrected)

        predicted_proba = calibrated_model.predict_proba(X_test)[:, 1]
        uncorrected_roc_auc = roc_auc_score(y_test.values, predicted_proba)

        # predicted_labels = calibrated_model.predict(X_test)
        # uncorrected_confusion_matrix = confusion_matrix(y_test.values, predicted_labels)
        # uncorrected_accuracy_score = accuracy_score(y_test.values, predicted_labels)
        #
        # if self.verbose >= 1:
        #     print("Test ROC AUC (not corrected): {}".format(uncorrected_roc_auc))
        #     print("Test Confusion matrix (not corrected): \n {} \n".format(uncorrected_confusion_matrix))
        #     print("Test accuracy (not corrected): \n {} \n".format(uncorrected_accuracy_score))

        return uncorrected_roc_auc

    def evaluate_corrected(self):

        data_y = self.train_data.drop(['got_go'], axis=1)
        data_lbld = data_y[~data_y["class"].isna()]

        corrected_train_data = self.augmented_data_lbld
        test_data = self.test_data

        X_train_corrected = corrected_train_data.iloc[:, :-2]
        y_train_corrected = corrected_train_data.iloc[:, -1]
        w_corrected = corrected_train_data.iloc[:, -2]

        test_data = test_data.dropna(subset=['class'])
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]

        model = self.classifiers['classifier_y']

        skf_5 = StratifiedKFold(5, random_state=1)
        calibrated_model = CalibratedClassifierCV(model, cv=skf_5, method='isotonic')

        # Corrected model evaluation
        calibrated_model.fit(X_train_corrected, y_train_corrected, sample_weight=w_corrected)

        predicted_proba = calibrated_model.predict_proba(X_test)[:, 1]
        corrected_roc_auc = roc_auc_score(y_test.values, predicted_proba)

        # predicted_labels = calibrated_model.predict(X_test)
        # corrected_accuracy_score = accuracy_score(y_test.values, predicted_labels)
        # corrected_confusion_matrix = confusion_matrix(y_test.values, predicted_labels)
        #
        # if self.verbose >= 1:
        #     print("Test ROC AUC (corrected): {}".format(corrected_roc_auc))
        #     print("Test Confusion matrix (corrected): \n {} \n".format(corrected_confusion_matrix))
        #     print("Test accuracy (corrected): \n {} \n".format(corrected_accuracy_score))

        return corrected_roc_auc