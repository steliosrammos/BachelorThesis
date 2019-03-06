# Imports
import h2o
import pandas as pd
import numpy as np

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score

from nonconformist.cp import TcpClassifier
from src.conformal.classifier_adapter import MyClassifierAdapter


## Build Classifier X

## Build Classifier Y
# calculate bias correction coefficients
# build h2o model with bias corr. coeff. as weights

## Run conformal framework with classifer Y

####################### TEST #######################
# Load data in
data = pd.read_csv("/Users/steliosrammos/Documents/Education/Maastricht/DKE-Year3/BachelorThesis/bachelor_thesis/data/processed/data_classifier_y_labld_weights.csv", sep=";")

X = data.iloc[:300, 0:-2]
y = data.iloc[:300, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Connect to h2o session
try:
    h2o.connect()
except:
    h2o.init()

# Create/load the underlying model

# sklearn model
# model = SVC(probability=True, gamma='auto')

# h2o model (uncalibrated)
# clf_y = h2o.load_model('/private/tmp/correct_classifier_y/Grid_GBM_py_5_sid_a301_model_python_1551863370174_1_model_5')

# h2o model (uncalibrated)
clf_y = h2o.load_model('/private/tmp/correct_calibrated_classifier_y/Grid_GBM_py_11_sid_a301_model_python_1551863370174_1816_model_1')

model = MyClassifierAdapter(clf_y)

# Create a default nonconformity function
# nc = ClassifierNc(corrected_clf_y)

# Create a transductive conformal classifier
if model is not None:
    tcp = TcpClassifier(model)
else:
    print('Failed to build tcp model.')

# Fit the TCP using the proper training set
tcp.fit(X_train, y_train)

prediction_probas = tcp.predict_conf(X_test.values)
####################### END TEST #######################