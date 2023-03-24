import pandas as pd
import pickle

#CITED FROM OPEN SOURCE PROJECT

# Tokenized notes whole
f = open('test_picklefiles/tokenized_notes.pckl', 'rb')
notes = pickle.load(f)
f.close()
df = pd.DataFrame(notes)
df.to_hdf('test_picklefiles/tokenized_notes.h5', key='df')

#Emebedding matrix for w2v whole
f = open('test_picklefiles/embedding_matrix_w2v.pckl', 'rb')
embedding_matrix = pickle.load(f)
f.close()
df = pd.DataFrame(embedding_matrix)
df.to_hdf('test_picklefiles/embedding_matrix_w2v.h5', key='df')

#Embedding matrix for gnv whole
f = open('test_picklefiles/embedding_matrix_GNV.pckl', 'rb')
embedding_matrix = pickle.load(f)
f.close()
df = pd.DataFrame(embedding_matrix)
df.to_hdf('test_picklefiles/embedding_matrix_GNV.h5', key='df')

#####################################################################################

# Tokenized notes eff
f = open('test_picklefiles/tokenized_notes_eff.pckl', 'rb')
notes = pickle.load(f)
f.close()
df = pd.DataFrame(notes)
df.to_hdf('test_picklefiles/tokenized_notes_eff.h5', key='df')

#Emebedding matrix for w2v eff
f = open('test_picklefiles/embedding_matrix_w2v_eff.pckl', 'rb')
embedding_matrix = pickle.load(f)
f.close()
df = pd.DataFrame(embedding_matrix)
df.to_hdf('test_picklefiles/embedding_matrix_w2v_eff.h5', key='df')

#Embedding matrix for gnv eff
f = open('test_picklefiles/embedding_matrix_GNV_eff.pckl', 'rb')
embedding_matrix = pickle.load(f)
f.close()
df = pd.DataFrame(embedding_matrix)
df.to_hdf('test_picklefiles/embedding_matrix_GNV_eff.h5', key='df')

#############################################################################################