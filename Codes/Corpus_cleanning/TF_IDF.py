import pandas as pd
import glob
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def create_list(i):
    files = os.listdir('doc_wordlist/')
    files.sort()
    
    for file in files:
        print(file)
        df = pd.read_pickle('doc_wordlist/'+file)
        df['normalized_text'] = df['normalized_tokens'].apply(lambda x: ' '.join(x))
        a = df['normalized_text'].values.reshape(-1).tolist()
        
        with open('list_doc/'+file, 'wb') as fp:
            pickle.dump(a, fp)
            
    return 0

def append_list(i):
    n_lst = [57, 57, 56, 56, 56]
    
    agg = []
    for i in range(5):
        for j in range(n_lst[i]):
            print(j)
            a = pd.read_pickle('list_doc/agg'+str(i+1)+'_'+str(j)+'.pkl')
            agg = agg+a
    
    with open('list_doc/agg.pkl', 'wb') as fp:
        pickle.dump(agg, fp)
    return 0

corpus = pd.read_pickle('list_doc/agg.pkl')

TFVectorizer = TfidfVectorizer(max_df=0.5, min_df=20, stop_words='english', norm='l2')

TFVects = TFVectorizer.fit_transform(corpus)

with open('TF_IDF/TFVects_05_40.pkl', 'wb') as fp:
    pickle.dump(TFVects, fp)
with open('TF_IDF/TFVectorizer_05_40.pkl', 'wb') as fp:
    pickle.dump(TFVectorizer, fp)