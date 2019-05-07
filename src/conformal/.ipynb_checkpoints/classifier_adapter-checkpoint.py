# General imports
import numpy as np

# Import h2o framework
import h2o

# Import conformist classes
from nonconformist.base import ClassifierAdapter
from nonconformist.nc import ClassificationErrFunc


class MyClassifierAdapter(ClassifierAdapter):

    def __init__(self, model, fit_params=None):
        super(MyClassifierAdapter, self).__init__(model, fit_params)
        self.normalizer = None
        self.err_func = MarginErrFunc()
        self.h2o_model = model

    def fit(self, x, y):
        '''
            x is a numpy.array of shape (n_train, n_features)
            y is a numpy.array of shape (n_train)

            Here, do what is necessary to train the underlying model
            using the supplied training data
        '''
        print('Using overwritten fit.')
        x = h2o.H2OFrame(x)
        y = h2o.H2OFrame(y)
        self.model.fit(x, y, verbose=False)

        return self.model

    def predict(self, x):
        '''
            Obtain predictions from the underlying model

            Make sure this function returns an output that is compatible with
            the nonconformity function used. For default nonconformity functions,
            output from this function should be class probability estimates in
            a numpy.array of shape (n_test, n_classes)
        '''

        predictions = self.h2o_model.predict(x, verbose=False)
        print(predictions)

        return predictions

    def score(self, x, y=None):
        """Calculates the nonconformity score of a set of samples.
        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of examples for which to calculate a nonconformity score.
        y : numpy array of shape [n_samples]
            Outputs of examples for which to calculate a nonconformity score.
        Returns
        -------
        nc : numpy array of shape [n_samples]
            Nonconformity scores of samples.
        """
        x = h2o.H2OFrame(x)
        y = h2o.H2OFrame(y)

        prediction = self.h2o_model.predict(x)
        n_test = x.shape[0]
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)
        #             print(norm)

        score = self.err_func.apply(prediction, y) / norm

        #         score = h2o.H2OFrame(score)
        #         print(score)

        return score


class MarginErrFunc(ClassificationErrFunc):
    """
    Calculates the margin error.
    For each correct output in ``y``, nonconformity is defined as
    .. math::
        0.5 - \dfrac{\hat{P}(y_i | x) - max_{y \, != \, y_i} \hat{P}(y | x)}{2}
    """

    def __init__(self):
        super(MarginErrFunc, self).__init__()

    def apply(self, prediction, y):
        prob = np.zeros(y.shape[0], dtype=np.float32)
        for i, y_ in enumerate(y):
            if y_ >= prediction.shape[1]:
                prob[i] = 0
            else:
                prob[i] = prediction[i, int(y_)]
                prediction[i, int(y_)] = -np.inf

        return 0.5 - ((prob - prediction.max()) / 2)
