# NLP_Smoking_Extraction
Natural Language Processing and Deep Learning in order to determine smoking status

NLP Pipeline for determining smoking status based on patient progress notes

Create virtual environment from requirements.txt

Pretrained word vectors must be downloaded from https://www.kaggle.com/datasets/anindya2906/glove6b

To extract file, from directory Cleaning and Processing run command: python -m gensim.scripts.glove2word2vec --input  glove.6B.100d.txt --output glove.6B.100d.w2vformat.txt

Run ReadSmokeData.py first to load and preprocess data, this will take a few minutes to complete

The various model files in Machine Learning Directory do not need to be run in any particular order, but must all be run before graphs can be generated

run our_graphs.py to generate graphs for this experiment
graphs from this experiment are stored in Our_Results folder within the Machine Learning directory

run graphs_from_paper.py to generate graphs with hardcoded values from reference paper for comparison purposes
graphs from reference paper data are stored in Paper_Results folder within the Machine Learning Directory

Directories Our_Results and Paper_Results already contain graphs that were produced during experiments and are included in report