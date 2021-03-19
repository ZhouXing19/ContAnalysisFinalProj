#All these packages need to be installed from pip
import gensim#For word2vec, etc
import requests #For downloading our datasets
import lucem_illud #pip install -U git+git://github.com/UChicago-Computational-Content-Analysis/lucem_illud.git

import math
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

import io


import pickle

import logging
logger = logging.getLogger("spacy")
logger.setLevel(logging.ERROR)

def ChangeCode(val):
    if math.isnan(val):
        return val
    return int(str(val)[:2])

def saveWordVecPairInFile(wv, sector):
  out_v = io.open(f'Data/tsvs/{sector}_vectors.tsv', 'w', encoding='utf-8')
  out_m = io.open(f'Data/tsvs/{sector}_metadata.tsv', 'w', encoding='utf-8')

  vocabulary = list(wv.vocab.keys())

  for index, word in enumerate(vocabulary):
     vec = wv[word]
     out_v.write('\t'.join([str(x) for x in vec]) + "\n")
     out_m.write(word + "\n")
  out_v.close()
  out_m.close()



dataPath = "./Data/MDA_2010_2020/doc_wordlist"
metaPath = "./Data/MDA_2010_2020/master/master_form10k.csv"
industryDataPath = "./Data/MDA_2010_2020/master_industry.csv"
industryDescriptionPath = "./Data/MDA_2010_2020/GICS_map_2018.xlsx"

# Load metadata
metadata = pd.read_csv(metaPath)

# Load Industry data
industryData = pd.read_csv(industryDataPath)
FName_gind = industryData[['FName', 'gind']]
FName_gind.gind = FName_gind.gind.apply(ChangeCode).astype('Int64')

# Load Industry description
industryDescData = pd.read_excel(industryDescriptionPath)
indData = industryDescData[industryDescData.columns[0:2]].dropna()
indData.columns = ['codes', 'indName']
CodeIndTable = indData.set_index('codes').to_dict()['indName']
IndCodeTable = {v: k for k, v in CodeIndTable.items()}
indCodes = set(CodeIndTable.keys())


# Convert the date to datetime
for col in ["FDATE", "FINDEXDATE", "LINDEXDATE"]:
    metadata[col] = metadata[col].astype(str)
    metadata[col] = pd.to_datetime(metadata[col], format="%Y%m%d", errors='coerce')


desiredIndustries = ['Information Technology', 'Financials', 'Energy', 'Materials']
desiredCodes = set([IndCodeTable[industry] for industry in desiredIndustries])

# Load text data, build the textDatadf
textDatadf = pd.DataFrame()
FilesNames = os.listdir(dataPath)
for idx, FileName in enumerate(FilesNames):
    FilePath = os.path.join(dataPath, FileName)
    curfile = open(FilePath, 'rb')
    curDf = pickle.load(curfile)
    # left join the file industry code
    curDf = pd.merge(left=curDf, right=FName_gind, how='left', left_on='FName', right_on='FName')
    filteredDf = curDf.loc[curDf['gind'].isin(desiredCodes)]
    textDatadf = textDatadf.append(filteredDf)
    if idx % 10 == 0:
        print(f"finished: {idx} , {FileName}")


df_list = np.array_split(textDatadf, 30)

for i in range(len(df_list)):
    this_seg_df = df_list[i]
    this_seg_df.to_pickle(f"./Data/MDA_2010_2020/textDatadf/textDatadf_{i}.pkl")
    if i % 5 == 0:
        print(f"finished for {i} sub_dfs")




# get word Embedding for each industry
# for industry in desiredIndustries:
#     code = IndCodeTable[industry]
#     this_ind_df = textDatadf[textDatadf.gind == code]
#     print(f'[{industry}] => {len(this_ind_df)} data')
#     senReleasesW2V = gensim.models.word2vec.Word2Vec(this_ind_df['normalized_tokens'].sum())
#     print(f' ===== finished for {industry} =====')
#     this_wv = senReleasesW2V.wv
#     saveWordVecPairInFile(this_wv, industry)
