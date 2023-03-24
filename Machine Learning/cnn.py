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
from keras.regularizers import l1
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



notes, embedding_matrix_w2v, embedding_matrix_GNV, word_index, max_len, notes_eff, embedding_matrix_w2v_eff, embedding_matrix_GNV_eff, word_index_eff, max_len_eff, binary_labels, categorical_labels = get_variables()

# Choose which setting you are running the model in and comment one othe two next lines out, go down to bottom of file and uncomment according test
#X_train_b, X_test_b, y_train_b, y_test_b, X_train_c, X_test_c, y_train_c, y_test_c = make_eff_labels(notes_eff, binary_labels, categorical_labels) #eff labels model
X_train_b, X_test_b, y_train_b, y_test_b, X_train_c, X_test_c, y_train_c, y_test_c = make_whole_labels(notes, binary_labels, categorical_labels) #whole labels model

# # temporary for local testing:
#X_train_b = X_train_b[:10]
#y_train_b = y_train_b[:10]
#X_test_b = X_test_b[:][:10]
#y_test_b = y_test_b[:][:10]

#X_train_c = X_train_c[:10]
#y_train_c = y_train_c[:10]
#X_test_c = X_test_c[:][:10]
#y_test_c = y_test_c[:][:10]

#print("Y_train b ", type(y_train_b))
#print("Y train c ", type(y_train_c))


#################################################################################################

#Model architecture cited from codebase
# CNN Model Version 2
def CNN_model2(word_index, embedding_matrix, max_len, categorical):
    #print("EMBED 2 SIZE ", len(word_index)+1)
    optm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 100, weights=[embedding_matrix], input_length=max_len, trainable=False))
    model.add(Conv1D(128, 7, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 7, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(Dropout(0.2))
    if (categorical):
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optm, metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optm, metrics=['accuracy'])

    return model

# Create CNN model
def CNN_model(word_index, embedding_matrix, max_len, categorical):
    #print("EMBED 1 SIZE ", len(word_index)+1)
    model = Sequential()
    optm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.add(Embedding(len(word_index)+1, 100, weights=[embedding_matrix], input_length=max_len, trainable=False)) #trainable=True for embedding matrix w2v 
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(10, activation='relu', activity_regularizer=l1(0.001)))
    model.add(Dropout(0.2))
    if (categorical):
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optm, metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optm, metrics=['accuracy'])
    
    return model

#################################################################################################

# CNN Model
def CNN(X_train, y_train, X_test, y_test, word_index, embedding_matrix, max_len, seed, categorical):
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto', restore_best_weights=True) # patience is number of epochs
    callbacks_list = [earlystop]
    if (categorical):
        kfold = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(X_train, y_train))
    else:
        kfold = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train, y_train))
    model_infos = []
    metrics = []
    model = None
    
    for i,(train, test) in enumerate(kfold):
        model = None
        #model = CNN_model(word_index, embedding_matrix, max_len, categorical)
        #print(y_train[train])
        model = CNN_model(word_index, embedding_matrix, max_len, categorical)
        print("Fit fold", i+1," ==========================================================================")
        model_info=model.fit(X_train[train], y_train[train], epochs=100, batch_size=8, validation_data=(X_train[test], y_train[test]),
                               callbacks=callbacks_list, verbose=1)
        print("Performance plot of fold {}:".format(i+1))
        # summarize history in plot
        #create_graphs(model_info) #ADD BACK IN TO PLOT
        model_infos.append(model_info)

        #Final evaluation of the model
        metrics, y_pred = evaluate_model(metrics, categorical, model, y_test, X_test)
    
    print(model.summary())
    
    return y_pred, metrics, model_infos

########################################################################################################


seed = 97

#Uncomment whichever model you would like to test

####################################################################################################################################
# Binary Tests untrained model, dropout 0.4
#y_pred, metrics, model_infos = CNN(X_train_b, y_train_b, X_test_b, y_test_b, word_index, embedding_matrix_GNV, max_len, seed, False)
#avg_metrics = findAverage(metrics)
#print("Average Scores")
#print(avg_metrics)

#create picklefile of binary results
#f = open('ML_picklefiles/CNN_binary_post.pckl', 'wb') #this is backwards, this is the pretrained model
#pickle.dump(avg_metrics, f)
#f.close()
#####################################################################################################################################


####################################################################################################################################
# Binary Tests pretrained model, dropout 0.2
#y_pred, metrics, model_infos = CNN(X_train_b, y_train_b, X_test_b, y_test_b, word_index, embedding_matrix_w2v, max_len, seed, False)
#avg_metrics = findAverage(metrics)
#print("Average Scores")
#print(avg_metrics)

#create picklefile of binary results
#f = open('ML_picklefiles/CNN_binary_pre.pckl', 'wb') #this is backwards, this is the post trained result
#pickle.dump(avg_metrics, f)
#f.close()
#####################################################################################################################################



##################################################################################################################################################
#Categorical tests pretrained model
y_pred, metrics, model_infos = CNN(X_train_c, y_train_c, X_test_c, y_test_c, word_index, embedding_matrix_w2v, max_len, seed, True)
avg_metrics = findAverage(metrics)
print("Average Scores")
print(avg_metrics)

#create picklefile of categorical results pretrained
#f = open('ML_picklefiles/CNN_categorical_pre.pckl', 'wb') #this is backward, this is the posttrained model
#pickle.dump(avg_metrics, f)
#f.close()
################################################################################################################################################

#################################################################################################################################################
#Categorical tests untrained model
#y_pred, metrics, model_infos = CNN(X_train_c, y_train_c, X_test_c, y_test_c, word_index, embedding_matrix_GNV, max_len, seed, True)
#avg_metrics = findAverage(metrics)
#print("Average Scores")
#print(avg_metrics)

#create picklefile of categorical results untrained
#f = open('ML_picklefiles/CNN_categorical_post.pckl', 'wb') #this is backward, this is the pretrained model
#pickle.dump(avg_metrics, f)
#f.close()
####################################################################################################################################################
