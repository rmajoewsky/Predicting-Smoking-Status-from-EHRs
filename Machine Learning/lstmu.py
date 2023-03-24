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

from keras.models import Sequential
from keras.layers import Dense, Activation, Input, LSTM, Embedding, Dropout, GRU, Bidirectional
from keras.layers import Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras import regularizers
from keras.layers import BatchNormalization
from keras.models import Model
from keras import metrics
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

from utils import get_variables, make_whole_labels, make_eff_labels, evaluate_model, findAverage

#sbatch --no-requeue : command to not repeat


notes, embedding_matrix_w2v, embedding_matrix_GNV, word_index, max_len, notes_eff, embedding_matrix_w2v_eff, embedding_matrix_GNV_eff, word_index_eff, max_len_eff, binary_labels, categorical_labels = get_variables()

# Choose which setting you are running the model in and comment one othe two next lines out
X_train_b, X_test_b, y_train_b, y_test_b, X_train_c, X_test_c, y_train_c, y_test_c = make_eff_labels(notes_eff, binary_labels, categorical_labels) #eff labels model
#X_train_b, X_test_b, y_train_b, y_test_b, X_train_c, X_test_c, y_train_c, y_test_c = make_whole_labels(notes, binary_labels, categorical_labels) #whole labels model

# # temporary for local testing:
#X_train_b = X_train_b[:10]
#y_train_b = y_train_b[:10]
#X_test_b = X_test_b[:][:10]
#y_test_b = y_test_b[:][:10]

#X_train_c = X_train_c[:10]
#y_train_c = y_train_c[:10]
#X_test_c = X_test_c[:][:10]
#y_test_c = y_test_c[:][:10]



#################################################################################################
# Create LSTM unidirectional model

#CITED MODEL ARCHITECTURE STRUCTURE FROM CODEBASE

def LSTM_Uni_model(word_index, embedding_matrix, max_len, categorical):
    optm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 100, weights=[embedding_matrix], input_length=max_len, trainable=True)) #trainable for w2v embedding matrix
    model.add(Dropout(0.4))    #DROPOUT
    model.add(Conv1D(filters=32, kernel_size=3, padding='same')) 
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(128))
    if (categorical):
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optm, metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optm, metrics=['accuracy'])
    
    return model

####################################################################################################

def LSTM_Uni(X_train, y_train, X_test, y_test, word_index, embedding_matrix, max_len, seed, categorical):
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto', restore_best_weights=True) # pateince is number of epochs
    callbacks_list = [earlystop]
    if (categorical):
        kfold = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(X_train, y_train))
    else:
        kfold = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train, y_train))
    model_infos = []
    metrics = []
    model = None
    for i,(train, test) in enumerate(kfold):
        #print("TRAIN ", train)
        model = None
        model = LSTM_Uni_model(word_index, embedding_matrix, max_len, categorical)
        print("Fit fold", i+1," ==========================================================================")
        model_info=model.fit(X_train[train], y_train[train], epochs=100, batch_size=8, validation_data=(X_train[test], y_train[test]),
                               callbacks=callbacks_list, verbose=1)
        print("Performance plot of fold {}:".format(i+1))
        # summarize history in plot
        #plot_model_history(model_info) ADD BACK IN FOR PLOTS
        model_infos.append(model_info)

        #Final evaluation of the model
        metrics, y_pred = evaluate_model(metrics, categorical, model, y_test, X_test)
    
    print(model.summary())
    
    return y_pred, metrics, model_infos

    ######################################################################################

seed = 97

#UNCOMMENT WHICHEVER VERSION OF MODEL YOU WOULD LIKE TO TEST, CHANGE PARAMETERS IN ACCORDANCE WITH SETTING SELECTED ABOVE

# Binary Tests
#y_pred, metrics, model_infos = LSTM_Uni(X_train_b, y_train_b, X_test_b, y_test_b, word_index_eff, embedding_matrix_GNV_eff, max_len_eff, seed, False)
#avg_metrics = findAverage(metrics)
#print("Average Scores")
#print(avg_metrics)

#create picklefile of binary results untrained
#f = open('ML_picklefiles/LSTMU_binary_post.pckl', 'wb') 
#pickle.dump(avg_metrics, f)
#f.close()

#create picklefile of binary results pretrained
#f = open('ML_picklefiles/LSTMU_binary_pre.pckl', 'wb') #Dropout 0.0
#pickle.dump(avg_metrics, f)
#f.close()


# Categorical Tests
y_pred, metrics, model_infos = LSTM_Uni(X_train_c, y_train_c, X_test_c, y_test_c, word_index_eff, embedding_matrix_w2v_eff, max_len_eff, seed, True)
avg_metrics = findAverage(metrics)
print("Average Scores")
print(avg_metrics)

#create picklefile of categorical results untrained
#f = open('ML_picklefiles/LSTMU_categorical.pckl', 'wb') 
#pickle.dump(avg_metrics, f)
#f.close()

