import nltk
import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import os
import csv
import pickle
from timeit import default_timer as timer
import inflect
from autocorrect import spell
from collections import OrderedDict
#import progressbar as pb

# function that cleans text
# still need to account for contractions, abbreviations, and numbers/fractions
default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english') # or any other list of your choice
def clean_text(text, remove_punctuation = False, remove_stopwords = False):

        def misc_cleaning(text):
                text = re.sub("-([a-zA-Z]+)", r"\1", text) # replaces hyphen with spaces in case of strings
                text = re.sub(' y ', '', text) # gets rid of random y accent stuff scattered through the text
                text = re.sub('yyy', 'y', text)
                text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
                text = re.sub(r"what's", "what is ", text)
                text = re.sub(r"\'s", " ", text)
                text = re.sub(r"\'ve", " have ", text)
                text = re.sub(r"can't", "cannot ", text)
                text = re.sub(r"n't", " not ", text)
                text = re.sub(r"i'm", "i am ", text)
                text = re.sub(r"\'re", " are ", text)
                text = re.sub(r"\'d", " would ", text)
                text = re.sub(r"\'ll", " will ", text)
                text = re.sub(r",", " ", text)
                text = re.sub(r"\.", " ", text)
                text = re.sub(r"!", " ! ", text)
                text = re.sub(r"\/", " ", text)
                text = re.sub(r"\^", " ^ ", text)
                text = re.sub(r"\+", " + ", text)
                text = re.sub(r"\-", " - ", text)
                text = re.sub(r"\=", " = ", text)
                text = re.sub(r"'", " ", text)
                text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
                text = re.sub(r":", " : ", text)
                text = re.sub(r" e g ", " eg ", text)
                text = re.sub(r" b g ", " bg ", text)
                text = re.sub(r" u s ", " american ", text)
                text = re.sub(r"\0s", "0", text)
                text = re.sub(r" 9 11 ", "911", text)
                text = re.sub(r"e - mail", "email", text)
                text = re.sub(r"j k", "jk", text)
                text = re.sub(r"\s{2,}", " ", text)
                return text

        # function to tokenize text which is used in a lot of the later processing
        def tokenize_text(text):
                return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

        text = text.strip(' ') # strip whitespaces
        text = misc_cleaning(text) # look at function, random cleaning stuff
        text = text.lower() # lowercase

        
        # removes punctuation
        if remove_punctuation:
                text = text.translate(str.maketrans('', '', string.punctuation))

       
        if remove_stopwords:
                stop_words = default_stopwords
                tokens = [w for w in tokenize_text(text) if w not in stop_words]
                text = " ".join(tokens)
        

        return text
