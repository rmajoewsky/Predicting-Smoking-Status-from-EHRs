from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim
from gensim.models import Word2Vec
import numpy as np
import pickle
from timeit import default_timer as timer
from nltk import word_tokenize
import statistics

f = open('test_picklefiles/cleaned_notes.pckl', 'rb')
notes = pickle.load(f)
f.close()
start = timer()

def initialize(text):
    t = Tokenizer()
    t.fit_on_texts(text) #training phase
    word_index = t.word_index #get a map of word index
    sequences = t.texts_to_sequences(text)
    return word_index, sequences

# another parameter was data --> avoiding that for now
def textTokenize(text):    
    word_index, sequences = initialize(text)
    #print(sequences.shape)
    #max_len=max(lengths(sequences))
    max_len = max(len(elem) for elem in sequences)
    print('Found %s unique tokens' % len(word_index))
    text_tok=pad_sequences(sequences, maxlen=max_len)
    return text_tok, word_index, max_len

def textTokenizeEff(text):
    word_index, sequences = initialize(text)
    lengths = [len(elem) for elem in sequences]
    mean = statistics.mean(lengths)
    std_dev = statistics.stdev(lengths)
    max_len = int((mean + 4*std_dev) + 1)
    print('Found %s unique tokens' % len(word_index))
    text_tok=pad_sequences(sequences, maxlen=max_len)
    return text_tok, word_index, max_len

def embed_matrices(word_index, embedding_index):
    #extract word embedding for train and test data
    embedding_matrix = np.zeros((len(word_index) + 1, 100))
    #print("EMBEDDING MATRIX.shape ", embedding_matrix.shape)
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
    #        print("embedding vector GNV.shape ", embedding_vector.shape)
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def word_Embed_GNV(word_index): 
    #This pretrained dataset has 100 dims, which is what our current method of reading in data is producing in the w2v model function  
    pretrain = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.100d.w2vformat.txt') #add binary=True for Google set  
    #convert pretrained word embedding to a dictionary
    embedding_index=dict()
    for i in range(len(pretrain.key_to_index)): #.wv doesn't work
        word=pretrain.index_to_key[i]
        if word is not None:
            embedding_index[word]=pretrain[word] 
    return embed_matrices(word_index, embedding_index) 
    
def make_w2v_model(notes, window, workers, epochs):
    model = gensim.models.Word2Vec(notes, sg=100, window=window, min_count=1, workers=workers)
    #print('Start training process...') 
    model.train(notes,total_examples=len(notes),epochs=epochs)
    model.save("w2v.model")
    print("Model Saved")


def word_Embed_w2v(word_index, model):   
    pretrain = model
    #convert pretrained word embedding to a dictionary
    embedding_index=dict()
    for i in range(len(pretrain.wv.key_to_index)): 
        word=pretrain.wv.index_to_key[i]
        if word is not None:
            embedding_index[word]=pretrain.wv[word]  
    #extract word embedding for train and test data
    return embed_matrices(word_index, embedding_index)
    
# Execution of variable creation for both whole and efficient
notes_tok, word_index, max_len = textTokenize(notes[2])
print("max_len of one patient's notes: ", max_len)
notes_tok_eff, word_index_eff, max_len_eff = textTokenizeEff(notes[2])
print("max_len of one patient's notes_eff: ", max_len_eff)
embedding_matrix_GNV = word_Embed_GNV(word_index)
embedding_matrix_GNV_eff = word_Embed_GNV(word_index_eff)
make_w2v_model(notes[2], 5, 10, 20)
w2v_model = Word2Vec.load("w2v.model")
embedding_matrix_w2v = word_Embed_w2v(word_index, w2v_model)
embedding_matrix_w2v_eff = word_Embed_w2v(word_index_eff, w2v_model)

#CITED FROM ORIGNAL PROJECT FOR CONSISTENCY
def pickle_whole_variable():
    f = open('test_picklefiles/tokenized_notes.pckl', 'wb')
    pickle.dump(notes_tok, f)
    f.close()
    print("Saved Tokenized Notes")

    f = open('test_picklefiles/embedding_matrix_GNV.pckl', 'wb')
    pickle.dump(embedding_matrix_GNV, f)
    f.close()
    print("Saved Google Vector Word Embedding Matrix")

    f = open('test_picklefiles/embedding_matrix_w2v.pckl', 'wb')
    pickle.dump(embedding_matrix_w2v, f)
    f.close()
    print("Saved Word 2 Vector Embedding Matrix")

    f = open('test_picklefiles/word_index.pckl', 'wb')
    pickle.dump(word_index, f)
    f.close()
    print("Saved Word Indices")

    f = open('test_picklefiles/max_len.pckl', 'wb')
    pickle.dump(max_len, f)
    f.close()
    print("Saved Maximum Length of One Patient's Notes")

def pickle_eff_variable():
    f = open('test_picklefiles/tokenized_notes_eff.pckl', 'wb')
    pickle.dump(notes_tok_eff, f)
    f.close()
    print("Saved Tokenized Notes")

    f = open('test_picklefiles/embedding_matrix_GNV_eff.pckl', 'wb')
    pickle.dump(embedding_matrix_GNV_eff, f)
    f.close()
    print("Saved Google Vector Word Embedding Matrix")

    f = open('test_picklefiles/embedding_matrix_w2v_eff.pckl', 'wb')
    pickle.dump(embedding_matrix_w2v_eff, f)
    f.close()
    print("Saved Word 2 Vector Embedding Matrix")

    f = open('test_picklefiles/word_index_eff.pckl', 'wb')
    pickle.dump(word_index_eff, f)
    f.close()
    print("Saved Word Indices")

    f = open('test_picklefiles/max_len_eff.pckl', 'wb')
    pickle.dump(max_len_eff, f)
    f.close()
    print("Saved Maximum Length of One Patient's Notes")

# Save variables
pickle_whole_variable()
pickle_eff_variable()

end = timer() # around 5 - 8 minutes to run the whole thing
print("Done within " + str(end-start) + " seconds")