import pickle
import pandas as pd
import numpy as np
from keras.models import Model
from keras import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

#CITATED METRICS FROM CODEBASE, UPDATED FOR FINAL DRAFT


def get_variables():
    # Reading tokenized notes from panda files
    df = pd.read_hdf('../Cleaning and Processing/test_picklefiles/tokenized_notes.h5')
    df.reset_index(drop = True)
    notes = df.values.tolist()

    # Reading word2vec word embedding matrix from Panda Files and converting to numpy array
    df = pd.read_hdf('../Cleaning and Processing/test_picklefiles/embedding_matrix_w2v.h5')
    df.reset_index(drop = True)
    embedding_matrix_w2v = df.to_numpy()

    # Reading Google word embedding matrix from Panda Files and converting to numpy array
    df = pd.read_hdf('../Cleaning and Processing/test_picklefiles/embedding_matrix_GNV.h5')
    df.reset_index(drop = True)
    embedding_matrix_GNV = df.to_numpy()

    # Reading word index from Pickle
    f = open('../Cleaning and Processing/test_picklefiles/word_index.pckl', 'rb')
    word_index = pickle.load(f)
    f.close()

    # Reading max length from Pickle
    f = open('../Cleaning and Processing/test_picklefiles/max_len.pckl', 'rb')
    max_len = pickle.load(f)
    f.close()

    # Reading tokenized notes eff from panda files
    df = pd.read_hdf('../Cleaning and Processing/test_picklefiles/tokenized_notes_eff.h5')
    df.reset_index(drop = True)
    notes_eff = df.values.tolist()

    # Reading word2vec word embedding matrix eff from Panda Files and converting to numpy array
    df = pd.read_hdf('../Cleaning and Processing/test_picklefiles/embedding_matrix_w2v_eff.h5')
    df.reset_index(drop = True)
    embedding_matrix_w2v_eff = df.to_numpy()

    # Reading Google word embedding matrix eff from Panda Files and converting to numpy array
    df = pd.read_hdf('../Cleaning and Processing/test_picklefiles/embedding_matrix_GNV_eff.h5')
    df.reset_index(drop = True)
    embedding_matrix_GNV_eff = df.to_numpy()

    # Reading word index eff from Pickle
    f = open('../Cleaning and Processing/test_picklefiles/word_index_eff.pckl', 'rb')
    word_index_eff = pickle.load(f)
    f.close()

    # Reading max length eff from Pickle
    f = open('../Cleaning and Processing/test_picklefiles/max_len_eff.pckl', 'rb')
    max_len_eff = pickle.load(f)
    f.close()

    # Reading binary labels
    f = open('../Cleaning and Processing/test_picklefiles/smoking_binary_labels.pckl', 'rb')
    binary_labels = pickle.load(f)
    f.close()

    # Reading categorical labels
    f = open('../Cleaning and Processing/test_picklefiles/smoking_categorical_labels.pckl', 'rb')
    categorical_labels = pickle.load(f)
    f.close()

    return notes, embedding_matrix_w2v, embedding_matrix_GNV, word_index, max_len, notes_eff, embedding_matrix_w2v_eff, embedding_matrix_GNV_eff, word_index_eff, max_len_eff, binary_labels, categorical_labels

def make_whole_labels(notes, binary_labels, categorical_labels):
    # Binary Labels
    
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(notes, binary_labels, test_size=0.33, random_state=39)
    
    X_train_b = np.array(X_train_b)
    X_test_b = np.array(X_test_b)

    # Categorical Labels
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(notes, categorical_labels, test_size=0.33, random_state=39)
    X_train_c = np.array(X_train_c)
    X_test_c = np.array(X_test_c)

    return X_train_b, X_test_b, y_train_b, y_test_b, X_train_c, X_test_c, y_train_c, y_test_c

def make_eff_labels(notes_eff, binary_labels, categorical_labels):
    # Binary Labels
    
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(notes_eff, binary_labels, test_size=0.33, random_state=39)
    X_train_b = np.array(X_train_b)
    X_test_b = np.array(X_test_b)

    # Categorical Labels
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(notes_eff, categorical_labels, test_size=0.33, random_state=39)
    X_train_c = np.array(X_train_c)
    X_test_c = np.array(X_test_c)

    return X_train_b, X_test_b, y_train_b, y_test_b, X_train_c, X_test_c, y_train_c, y_test_c


def evaluate_model(metrics, categorical, model, y_test, X_test):
    y_pred=model.predict(X_test,verbose=1)
    #print("y_pred ", y_pred)
    if (categorical): ##Check this out, weird##
        y_pred_coded = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)
        metric=[]
        metric.append(['f1score',f1_score(y_test,y_pred_coded, average='weighted')])
        metric.append(['precision',precision_score(y_test,y_pred_coded, average='weighted')])
        metric.append(['recall',recall_score(y_test,y_pred_coded, average='weighted')])
        metric.append(['accuracy',accuracy_score(y_test,y_pred_coded)])
        print(metric)
        metrics.append(metric)
    else:
        y_pred_coded=np.where(y_pred>0.5,1,0)
        y_pred_coded=y_pred_coded.flatten()
        metric=[]
        metric.append(['f1score',f1_score(y_test,y_pred_coded)])
        metric.append(['precision',precision_score(y_test,y_pred_coded, labels=np.unique(y_pred_coded))])
        metric.append(['recall',recall_score(y_test,y_pred_coded)])
        metric.append(['accuracy',accuracy_score(y_test,y_pred_coded)])
        print(metric)
        metrics.append(metric)
    
    return metrics, y_pred

def findAverage(all_metrics):
    f1 = []
    precision = []
    recall = []
    accuracy = []
    avg_metrics = []
    for metrics in all_metrics:
        print(metrics)
        f1.append(metrics[0][1])
        precision.append(metrics[1][1])
        recall.append(metrics[2][1])
        accuracy.append(metrics[3][1])
    avg_metrics.append(['f1score',np.mean(f1)])
    avg_metrics.append(['precision',np.mean(precision)])
    avg_metrics.append(['recall',np.mean(recall)])
    avg_metrics.append(['accuracy',np.mean(accuracy)])
    
    return avg_metrics

