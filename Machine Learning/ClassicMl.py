import numpy as np
import tensorflow as tf
import random as rn
import os
import random
import statistics as st
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

# Reading in text
cleaned_notes = pd.read_pickle('../Cleaning and Processing/test_picklefiles/cleaned_notes.pckl')

# Reading binary labels
binary_labels = pd.read_pickle('../Cleaning and Processing/test_picklefiles/smoking_binary_labels.pckl')

# Reading categorical labels
categorical_labels = pd.read_pickle('../Cleaning and Processing/test_picklefiles/smoking_categorical_labels.pckl')

score_dict = {}
    
#print(categorical_labels.head())

# pull the data into vectors
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(cleaned_notes[2])

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(x, binary_labels, test_size=0.33, random_state = 39)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(x, categorical_labels, test_size=0.33, random_state=39)
y_train_c = np.argmax(y_train_c, axis=1)
y_test_c = np.argmax(y_test_c, axis=1)

samples = 1

###############################################################################
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer # add reference
print("Naive Bayes Binary Classification")

for i in range(0, samples):
    binary_naive_bayes = {}
    nb = MultinomialNB(alpha=1.0)
  
    nb.fit(X_train_b, y_train_b)
    y_pred = nb.predict(X_test_b)
    print('f1 Score %s' % f1_score(y_pred, y_test_b))
    binary_naive_bayes['f1 score'] = f1_score(y_pred, y_test_b)
    print('precision %s' % precision_score(y_pred, y_test_b))
    binary_naive_bayes['precision'] = precision_score(y_pred, y_test_b)
    print('recall %s' % recall_score(y_pred, y_test_b))
    binary_naive_bayes['recall'] = recall_score(y_pred, y_test_b)
    print('accuracy %s' % accuracy_score(y_pred, y_test_b))
    binary_naive_bayes['accuracy'] = accuracy_score(y_pred, y_test_b)
    print("---------------------------------")
    nb = None
    score_dict['binary_naive_bayes'] = binary_naive_bayes
    

print("####################################")
# ###############################################################################
from sklearn.linear_model import SGDClassifier
print("Binary Classification for SVM Model")

for i in range(0, samples):
    binary_svm = {}
    sgd = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=39, max_iter=5, tol=None)
    sgd.fit(X_train_b, y_train_b)
    y_pred = sgd.predict(X_test_b)
    print('f1 Score %s' % f1_score(y_pred, y_test_b))
    binary_svm['f1 score'] = f1_score(y_pred, y_test_b)
    print('precision %s' % precision_score(y_pred, y_test_b))
    binary_svm['precision'] = precision_score(y_pred, y_test_b)
    print('recall %s' % recall_score(y_pred, y_test_b))
    binary_svm['recall'] = recall_score(y_pred, y_test_b)
    print('accuracy %s' % accuracy_score(y_pred, y_test_b))
    binary_svm['accuracy'] = accuracy_score(y_pred, y_test_b)
    print("---------------------------------")
    sgd = None
    score_dict['binary_svm'] = binary_svm

print("####################################")
#################################################################################
from sklearn.linear_model import LogisticRegression
print("Binary Classification for Logistic Regression")

for i in range(0, samples):
    binary_logreg = {}
    logreg = LogisticRegression(n_jobs=1, C=1e5)
    logreg.fit(X_train_b, y_train_b)
    y_pred = logreg.predict(X_test_b)
    print('f1 Score %s' % f1_score(y_pred, y_test_b))
    binary_logreg['f1 score'] = f1_score(y_pred, y_test_b)
    print('precision %s' % precision_score(y_pred, y_test_b))
    binary_logreg['precision'] = precision_score(y_pred, y_test_b)
    print('recall %s' % recall_score(y_pred, y_test_b))
    binary_logreg['recall'] = recall_score(y_pred, y_test_b)
    print('accuracy %s' % accuracy_score(y_pred, y_test_b))
    binary_logreg['accuracy'] = accuracy_score(y_pred, y_test_b)
    print("---------------------------------")
    logreg = None
    score_dict['binary_logreg'] = binary_logreg

print("####################################")
##################################################################################
##################################################################################
print("Non-Binary Classification for Naive Bayes Model")

for i in range(0, samples):
    multi_nb = {}
    nb = MultinomialNB()
    nb.fit(X_train_c, y_train_c)
    y_pred = nb.predict(X_test_c)
    print('f1 Score %s' % f1_score(y_pred, y_test_c, average='weighted'))
    multi_nb['f1 score'] = f1_score(y_pred, y_test_c, average='weighted')
    print('precision %s' % precision_score(y_pred, y_test_c, average='weighted'))
    multi_nb['precision'] = precision_score(y_pred, y_test_c, average='weighted')
    print('recall %s' % recall_score(y_pred, y_test_c, average='weighted', labels=np.unique(y_pred)))
    multi_nb['recall'] = recall_score(y_pred, y_test_c, average='weighted', labels=np.unique(y_pred))
    print('accuracy %s' % accuracy_score(y_pred, y_test_c))
    multi_nb['accuracy'] = accuracy_score(y_pred, y_test_c)
    print("---------------------------------")
    nb = None
    score_dict['multi_nb'] = multi_nb


print("####################################")
##################################################################################
print("Non-Binary Classification for SVM Model")

for i in range(0, samples):
    multi_svm = {}
    sgd = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
    sgd.fit(X_train_c, y_train_c)
    y_pred = sgd.predict(X_test_c)
    print('f1 Score %s' % f1_score(y_pred, y_test_c, average='weighted'))
    multi_svm['f1 score'] = f1_score(y_pred, y_test_c, average='weighted')
    print('precision %s' % precision_score(y_pred, y_test_c, average='weighted'))
    multi_svm['precision'] = precision_score(y_pred, y_test_c, average='weighted')
    print('recall %s' % recall_score(y_pred, y_test_c, average='weighted'))
    multi_svm['recall'] = recall_score(y_pred, y_test_c, average='weighted')
    print('accuracy %s' % accuracy_score(y_pred, y_test_c))
    multi_svm['accuracy'] = accuracy_score(y_pred, y_test_c)
    print("---------------------------------")
    sgd = None
    score_dict['multi_svm'] = multi_svm

print("####################################")
##################################################################################
print("Non-Binary Classification for Logistic Regression Model")

for i in range(0, samples):
    multi_logreg = {}
    logreg = LogisticRegression(n_jobs=1, C=1e5)
    logreg.fit(X_train_c, y_train_c)
    y_pred = logreg.predict(X_test_c)
    print('f1 Score %s' % f1_score(y_pred, y_test_c, average='weighted'))
    multi_logreg['f1 score'] = f1_score(y_pred, y_test_c, average='weighted')
    print('precision %s' % precision_score(y_pred, y_test_c, average='weighted'))
    multi_logreg['precision'] = precision_score(y_pred, y_test_c, average='weighted')
    print('recall %s' % recall_score(y_pred, y_test_c, average='weighted'))
    multi_logreg['recall'] = recall_score(y_pred, y_test_c, average='weighted')
    print('accuracy %s' % accuracy_score(y_pred, y_test_c))
    multi_logreg['accuracy'] = accuracy_score(y_pred, y_test_c)
    print("---------------------------------")
    logreg = None
    score_dict['multi_logreg'] = multi_logreg

print("####################################")
##################################################################################

#create picklefile of dict of results
f = open('ML_picklefiles/ClassicML_scores.pckl', 'wb')
pickle.dump(score_dict, f)
f.close()