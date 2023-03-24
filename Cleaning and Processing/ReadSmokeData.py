# import numpy as np
import os
import pandas as pd
import xml.etree.ElementTree as Xet
from lxml import etree
import numpy as np
import pickle
from keras.utils import np_utils
import nltk
import xmltodict
import gensim
from gensim.models import Word2Vec
import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import inflect
from autocorrect import spell
from collections import OrderedDict
import progressbar as pb
from CleanNotes import clean_text

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


#uncomment below if needed
#nltk.download('punkt') 
#nltk.download('stopwords') 

#DESIRED_CLASSES_SS = set(["non_smoker","current_smoker","past_smoker","smoker","unknown","ever_smoker"])
smoking_binary_labels = []
smoking_categorical_labels = []

def make_binary_labels(word_label):
    labels = []
    
    if (word_label == 'PAST SMOKER' or word_label == 'CURRENT SMOKER' or word_label == 'SMOKER' or word_label == 'EVER SMOKER'): 
                #labels.append(1)
                return 1
    elif (word_label == 'NON-SMOKER' or word_label == 'UNKNOWN'):
            return 0
    else:
        return 2

    #labels = np.array(binary_labels)
    #return labels
    #print(word_labels)

def make_categorical_labels(word_label):
    #labels = []

    
    if (word_label == 'CURRENT SMOKER' or word_label == 'SMOKER'): 
                #labels.append(1)
                return 2
    elif (word_label == 'PAST SMOKER' or word_label == 'EVER SMOKER'):
        return 1
    elif (word_label == 'NON-SMOKER' or word_label == 'UNKNOWN'):
            return 0



filepath = 'testdata2/smoking_label_train/smokers_surrogate_train_all_version2.xml'
file_names = ["smokers_surrogate_test_all_groundtruth_version2.xml","smokers_surrogate_train_all_version2.xml"]

parser = etree.XMLParser(recover=True)
xmlparse = Xet.parse(filepath, parser)
root = xmlparse.getroot()

file_path = 'testdata2/smoking_readin/'

status = []
binary_list = []
categorical_list = []
for file_name in file_names:
    file = file_path + file_name
    with open(file) as fd:
        XML = xmltodict.parse(fd.read())
        idx = 0
        for key in XML["ROOT"]["RECORD"]:
            idx += 1
            #print(key, idx)

            patient_id = key["@ID"]
            answer_class = key["SMOKING"]["@STATUS"]
            smoking_binary_labels.append(make_binary_labels(answer_class)) #make binary labels
            smoking_categorical_labels.append(make_categorical_labels(answer_class)) #make categorical labels
            patient_note = key["TEXT"]
            patient_note = clean_text(patient_note, remove_punctuation = True, remove_stopwords = True)
            status.append([patient_id,answer_class,patient_note])

#Makes dataframe of 3 columns: ID--0, Smoking Status--1, and cleaned text--2
smoking_df = pd.DataFrame(status)
#binary_df = pd.DataFrame(smoking_binary_labels)
np_binary = np.array(smoking_binary_labels)
#print(smoking_df.head())

#make one-hot vector
labels = np.array(smoking_categorical_labels)
l = np_utils.to_categorical(labels)
#categorical_df = pd.DataFrame(l)
#print(l)
f = open('test_picklefiles/smoking_binary_labels.pckl', 'wb')
pickle.dump(np_binary, f)
#binary_labels = pickle.load(f)
f.close()

f = open('test_picklefiles/smoking_categorical_labels.pckl', 'wb')
pickle.dump(l, f)
f.close()

f = open('test_picklefiles/cleaned_notes.pckl', 'wb')
pickle.dump(smoking_df, f)
f.close()

#creates pickle and hdf files
exec(open("./WordEmbed.py").read())
exec(open("./toHdf.py").read())















