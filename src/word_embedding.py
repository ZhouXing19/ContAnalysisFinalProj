#All these packages need to be installed from pip
import gensim#For word2vec, etc
import requests #For downloading our datasets
import lucem_illud #pip install -U git+git://github.com/UChicago-Computational-Content-Analysis/lucem_illud.git

import numpy as np #For arrays
import pandas as pd #Gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import seaborn #Makes the graphics look nicer
import sklearn.metrics.pairwise #For cosine similarity
import sklearn.manifold #For T-SNE
import sklearn.decomposition #For PCA
import string

import os #For looking through files
import os.path #For managing file paths


import pickle

import logging
logger = logging.getLogger("spacy")
logger.setLevel(logging.ERROR)

dataPath = "./Data/MDA_2010_2020/mda_agg/mda_agg4.pkl"
metaPath = "./Data/MDA_2010_2020/master/master_form10k.csv"

# Load metadata
metadata = pd.read_csv(metaPath)

# Convert the date to datetime
for col in ["FDATE", "FINDEXDATE", "LINDEXDATE"]:
    metadata[col] = metadata[col].astype(str)
    metadata[col] = pd.to_datetime(metadata[col], format="%Y%m%d", errors='coerce')

# Load text data
with open(dataPath, "rb") as input_file:
    mda_file = pickle.load(input_file)




def clean_str(this_str):
    this_str = this_str.lower()
    this_str = " ".join(this_str.split())
    return this_str

mda_file['mda'] = mda_file['mda'].apply(lambda x: clean_str(x))

#Apply our functions, notice each row is a list of lists now
mda_file['tokenized_sents'] = mda_file['mda'].apply(lambda x: [lucem_illud.word_tokenize(s) for s in lucem_illud.sent_tokenize(x)])
#senReleasesDF['normalized_sents'] = senReleasesDF['tokenized_sents'].apply(lambda x: [lucem_illud_2020.normalizeTokens(s, lemma=False) for s in x])
mda_file['normalized_sents'] = mda_file['tokenized_sents'].apply(lambda x: [lucem_illud.normalizeTokens(s) for s in x])

# mda_file.to_pickle("./Data/MDA_2010_2020/mda_agg/mda_agg1_token_nor.pkl")

senReleasesW2V = gensim.models.word2vec.Word2Vec(mda_file['normalized_sents'].sum())

words = mda_file['normalized_sents'].sum()

senReleasesW2V.wv.syn0