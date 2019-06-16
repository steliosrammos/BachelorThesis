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
        self.augmented_train_data = None

        self.classifiers = classifiers
        self.clf_parameters = clf_parameters
        self.rebalancing_parameters = rebalancing_parameters
        self.bias_correction_parameters = bias_correction_parameters
        self.verbose = verbose
        self.random_state = random_state

        negative = train_data[train_data['got_go'] == 0].shape[0]
        positive = train_data[train_data['got_go'] == 1].shape[0]
        self.class_ratio = (negative+1)/(positive+1)

    def compute_correction_weights(self):

        print('Start computing weights W')

        data_s = self.train_data.drop(['class', 'weight'], axis=1)

        # Shuffle rows
        X = data_s.drop('got_go', axis=1)
        y = data_s.loc[:, 'got_go']

        # Startified k-fold
        skf = StratifiedKFold(n_splits=10, random_state=self.random_state)

        prob_s_pos = y.loc[y == 1].shape[0] / y.shape[0]

        # predicted_prob_s = pd.Series(np.full(y.shape[0], np.nan))
        # predicted_prob_s.index = y.index

        fold_num = 0

        for train_index, valid_index in skf.split(X, y):

            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            # Compute weights for test examples
            clf_s = self.classifiers['classifier_s']
            clf_s.fit(X_train, y_train)

            predicted_prob_s = clf_s.predict_proba(X_valid)[:, 1]
            predicted_prob_s = pd.Series(predicted_prob_s)
            predicted_prob_s.index = X_valid.index

            for index in X_valid.index:
                # if predicted_prob_s.loc[index] > 0:
                self.train_data.loc[index, 'weight'] = prob_s_pos / predicted_prob_s.loc[index]
                # else:
                #     self.train_data.loc[index, 'weight'] = 0.001 / prob_s_pos
            fold_num += 1

        self.augmented_train_data = self.train_data

        data_y = self.train_data.drop('got_go', axis=1)
        data_lbld = data_y[~data_y['class'].isna()]
        self.augmented_data_lbld = data_lbld

        return True

    def update_correction_weights(self, data_lbld, data_unlbld):

        # print('Updating weights W...')

        data_s = self.train_data.drop(['class', 'weight'], axis=1)
        data_s.loc[data_lbld.index.values, 'got_go'] = 1
        data_s.loc[data_unlbld.index.values, 'got_go'] = 0

        # Shuffle rows
        X = data_s.drop('got_go', axis=1)
        y = data_s.loc[:, 'got_go']

        # Startified k-fold
        sss = StratifiedKFold(n_splits=10, random_state=self.random_state)

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
            clf_s = self.classifiers['classifier_s']
            clf_s.fit(X_train, y_train)

            predicted_prob_s.loc[X_valid.index] = clf_s.predict_proba(X_valid)[:, 1]

            for index in X_valid.index:
                if predicted_prob_s.loc[index] > 0:
                    data_s.loc[index, 'weight'] = prob_s_pos / predicted_prob_s.loc[index]
                else:
                    data_s.loc[index, 'weight'] = 0.001 / prob_s_pos
            fold_num += 1

        data_lbld.weight = data_s.loc[data_lbld.index, 'weight']
        # data_unlbld.weight = data_s.loc[data_unlbld.index, 'weight']

        return data_lbld

    def visualize_weights(self):
        data = self.augmented_data_lbld
        weights = data.weight

        plt.hist(weights, bins=100)
        plt.yscale('log')
        plt.show()

    # Classical semi-supervised learning
    def classic_correct(self, percent_labeled):

        data_y = self.train_data.drop(['got_go'], axis=1)

        # Split data into labeled and unlabeled sets
        data_lbld = data_y[~data_y['class'].isna()]

        data_unlbld = data_y[data_y['class'].isna()]
        total_unlbld = data_unlbld.shape[0]
        new_labeled = 0
        cnt_labeled = 0

        stop = False

        while not stop:
            threshold = 0.96

            predictions = self.classic_predict(data_lbld, data_unlbld)

            filtered_negatives = predictions[predictions['class_0'] > threshold]
            filtered_positives = predictions[predictions['class_1'] > threshold]

            negative_indeces = list(filtered_negatives.index.values)
            positive_indeces = list(filtered_positives.index.values)

            # Keep class balance
            current_ratio = (len(negative_indeces)+1) / (len(positive_indeces)+1)

            if current_ratio < self.class_ratio:
                while current_ratio < self.class_ratio and len(positive_indeces) > 1:
                    positive_indeces = positive_indeces[1:]
                    current_ratio = (len(negative_indeces) + 1) / (len(positive_indeces) + 1)

            elif current_ratio > self.class_ratio:
                while current_ratio > self.class_ratio and len(negative_indeces) > 1:
                    negative_indeces = negative_indeces[1:]
                    current_ratio = (len(negative_indeces) + 1) / (len(positive_indeces) + 1)

            # print("Negative labels: {} \n Positive labels: {} \n".format(len(negative_indeces), len(positive_indeces)))

            data_unlbld.loc[negative_indeces, 'class'] = 0
            data_unlbld.loc[positive_indeces, 'class'] = 1

            newly_labeled = data_unlbld[~data_unlbld['class'].isna()]
            new_labeled += newly_labeled.shape[0]

            if (len(positive_indeces) == 0 or len(negative_indeces) == 0) or data_unlbld.shape[0] == 0 or new_labeled > total_unlbld * percent_labeled:
                print('Total labeled: {} \n'.format(cnt_labeled))
                self.augmented_data_lbld = data_lbld
                stop = True
            else:
                data_lbld = data_lbld.append(newly_labeled)
                data_unlbld = data_unlbld[data_unlbld['class'].isna()]

                if self.bias_correction_parameters['dynamic_weights']:
                    data_lbld = self.update_correction_weights(data_lbld, data_unlbld)

                cnt_labeled += newly_labeled.shape[0]

    def classic_predict(self, data_lbld, data_unlbld):

        # Create instance of classifier
        classifier_y = self.classifiers['classifier_y']
        parameters_y = self.clf_parameters['classifier_y']

        clf = classifier_y.set_params(**parameters_y)

        X = data_lbld.drop(['class', 'weight'], axis=1)
        y = data_lbld.loc[:, 'class']
        w = data_lbld.loc[:, 'weight']

        X_unlbld = data_unlbld.drop(['class', 'weight'], axis=1)

        clf.fit(X, y, sample_weight=w)

        predictions = clf.predict_proba(X_unlbld)
        predictions = pd.DataFrame(predictions, columns=['class_0', 'class_1'])
        predictions.index = data_unlbld.index

        return predictions

    def ccp_correct(self, percent_labeled):

        # Initialize stopping variable
        stop = False

        data_y = self.train_data.drop(['got_go'], axis=1)

        data_lbld = data_y[~data_y['class'].isna()]
        data_unlbld = data_y[data_y['class'].isna()]

        total_unlbld = data_unlbld.shape[0]
        cnt_labeled = 0
        new_labeled = 0

        while not stop:

            # Make conformal predictions
            ccp_predictions = self.ccp_predict(data_lbld, data_unlbld)

            # Get best prediction indeces and values
            labels = self.get_best_pred_indeces(ccp_predictions, 0.96)

            # Update counters
            new_labeled += labels.shape[0]
            remain_unlbld = data_unlbld.shape[0]-new_labeled

            if remain_unlbld <= 0 or labels.shape[0] == 0 or new_labeled > total_unlbld * percent_labeled:
                print('Total labeled: {} \n'.format(cnt_labeled))
                self.augmented_data_lbld = data_lbld
                stop = True
            else:
                # Add best instances to the labeled set
                data_unlbld.loc[labels.index.values, 'class'] = labels.values
                data_lbld = data_lbld.append(data_unlbld.loc[labels.index.values])

                # Drop those instances from the unlabeled set
                data_unlbld = data_unlbld.drop(labels.index.values)

                if self.bias_correction_parameters['dynamic_weights']:
                    data_lbld = self.update_correction_weights(data_lbld, data_unlbld)

                # Update confirmed additions
                cnt_labeled += labels.shape[0]

    def ccp_predict(self, data_lbld, data_unlbld):

        # Create SMOTE instance for class rebalancing
        smote = SMOTE(random_state=self.random_state)

        # Create instance of classifier
        classifier_y = self.classifiers['classifier_y']
        parameters_y = self.clf_parameters['classifier_y']

        clf = classifier_y.set_params(**parameters_y)

        X = data_lbld.drop(['class', 'weight'], axis=1)
        y = data_lbld.loc[:, 'class']
        w = data_lbld.loc[:, 'weight']

        X_unlbld = data_unlbld.drop(['class', 'weight'], axis=1)

        skf = StratifiedKFold(n_splits=5, random_state=self.random_state)

        p_values = []

        for train_index, calib_index in skf.split(X, y):

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
        ccp_predictions['credibility'] = [row.max() for _, row in ccp_predictions.iterrows()]
        ccp_predictions['confidence'] = [1-row.min() for _, row in ccp_predictions.iterrows()]
        ccp_predictions.index = X_unlbld.index

        return ccp_predictions

    @staticmethod
    def calculate_ratio(data_lbld, new_lbld):
        data_positives = data_lbld[data_lbld['class'] == 1].shape[0]
        data_negatives = data_lbld[data_lbld['class'] == 0].shape[0]

        new_positives = new_lbld[new_lbld['class'] == 1].shape[0]
        new_negatives = new_lbld[new_lbld['class'] == 0].shape[0]

        positives = data_positives + new_positives
        negatives = data_negatives + new_negatives

        ratio = negatives / positives

        return ratio

    def  get_best_pred_indeces(self, predictions, threshold):
        '''
            Returns the predictions that have a confidence level above a given threshold.
            The labels returned will have the same positive/negative ratio as the ratio specified by 'true_ratio'.
            true_ratio: num_negatives / num_positives
        '''

        predictions['labels'] = [np.argmax(row.values[0:2]) for _, row in predictions.iterrows()]

        positives = predictions[predictions['labels'] == 1].sort_values('confidence')
        negatives = predictions[predictions['labels'] == 0].sort_values('confidence')

        positives = positives.loc[(positives['confidence'] >= threshold) & (positives['credibility'] <= threshold)]
        negatives = negatives.loc[(negatives['confidence'] >= threshold) & (negatives['credibility'] <= threshold)]

        current_ratio = (negatives.shape[0] + 1) / (positives.shape[0] + 1)

        if current_ratio < self.class_ratio:
            while current_ratio < self.class_ratio and positives.shape[0] > 1:
                positives = positives[1:]
                current_ratio = (negatives.shape[0] + 1) / (positives.shape[0] + 1)

        elif current_ratio > self.class_ratio:
            while current_ratio > self.class_ratio and negatives.shape[0] > 1:
                negatives = negatives[1:]
                current_ratio = (negatives.shape[0] + 1) / (positives.shape[0] + 1)

        positives_indeces = list(positives.index.values)
        negatives_indeces = list(negatives.index.values)

        if self.verbose >= 2:
            print('Good positive predictions: {} \n'.format(len(positives_indeces)))
            print('Good negative predictions: {} \n'.format(len(negatives_indeces)))

        best_pred_indeces = positives_indeces + negatives_indeces

        # Terminate if one of two is empty
        if len(positives_indeces) == 0 or len(negatives_indeces) == 0:
            best_pred_indeces = []

        labels = predictions.loc[best_pred_indeces, 'labels']

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

        data_y = self.train_data.drop(['got_go', 'weight'], axis=1)
        uncorrected_train_data = data_y[~data_y['class'].isna()]

        X_train_uncorrected = uncorrected_train_data.drop('class', axis=1)
        y_train_uncorrected = uncorrected_train_data.loc[:, 'class']

        X_test = self.test_data.drop(['got_go', 'class', 'weight'], axis=1)
        y_test = self.test_data.loc[:, 'class']

        model = self.classifiers['classifier_y']

        skf_5 = StratifiedKFold(5)
        calibrated_model = CalibratedClassifierCV(model, cv=skf_5, method='isotonic')

        # Uncorrected model evaluation
        calibrated_model.fit(X_train_uncorrected, y_train_uncorrected)

        predicted_proba = calibrated_model.predict_proba(X_test)[:, 1]
        uncorrected_roc_auc = roc_auc_score(y_test.values, predicted_proba)

        print("Uncorrected ROC: ", uncorrected_roc_auc)
        return uncorrected_roc_auc

    def evaluate_corrected(self):

        corrected_train_data = self.augmented_data_lbld

        X_train_corrected = corrected_train_data.drop(['class', 'weight'], axis=1)
        y_train_corrected = corrected_train_data.loc[:, 'class']
        w_corrected = corrected_train_data.loc[:, 'weight']

        X_test = self.test_data.drop(['class', 'got_go', 'weight'], axis=1)
        y_test = self.test_data.loc[:, 'class']

        model = self.classifiers['classifier_y']
        skf_5 = StratifiedKFold(5)
        calibrated_model = CalibratedClassifierCV(model, cv=skf_5, method='isotonic')

        # Corrected model evaluation
        calibrated_model.fit(X_train_corrected, y_train_corrected, sample_weight=w_corrected)

        predicted_proba = calibrated_model.predict_proba(X_test)[:, 1]
        corrected_roc_auc = roc_auc_score(y_test.values, predicted_proba)

        print("Corrected ROC: {} \n".format(corrected_roc_auc))
        return corrected_roc_auc
