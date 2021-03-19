import pandas as pd
import gensim
import pickle

def create_dictionary(i):
    nfile_lst = [57, 57, 56, 56, 56]

    df = pd.DataFrame()
    for j in range(nfile_lst[i-1]):
        print(j)
    df = df.append(pd.read_pickle('doc_wordlist/agg'+str(i)+'_'+str(j)+'.pkl'))

    dictionary_r = gensim.corpora.Dictionary(df['normalized_tokens'])

    with open('dictionary_' + str(i) + '.pkl', 'wb') as fp:
        pickle.dump(dictionary_r, fp)
    
    print('success')
    return 0

with open('dictionary_' + str(1) + '.pkl', 'rb') as fp:
    dictionary_r = pickle.load(fp)

for i in range(2, 6):
    with open('dictionary_' + str(i) + '.pkl', 'rb') as fp:
        dict2 = pickle.load(fp)
    dictionary_r.merge_with(dict2)
    
with open('dictionary_r.pkl', 'wb') as fp:
    pickle.dump(dictionary_r, fp)
    
def save_bow(i):
    with open('dictionary_' + str(1) + '.pkl', 'rb') as fp:
        dictionary_r = pickle.load(fp)

#     dictionary_r.filter_extremes(no_below=10, no_above=0.5)

    nfile_lst = [57, 57, 56, 56, 56]

    for j in range(nfile_lst[i-1]):
        df = pd.read_pickle('doc_wordlist/agg'+str(i)+'_'+str(j)+'.pkl')

        corpus_r = [dictionary_r.doc2bow(text) for text in df['normalized_tokens']]

        with open('bow/bow'+str(i)+'_'+str(j)+'.pkl', 'wb') as fp:
            pickle.dump(corpus_r, fp)

    return 0

corpus_r = []
nfile_lst = [57, 57, 56, 56, 56]
for i in range(1, 6):
    for j in range(nfile_lst[i-1]):
        bow = pd.read_pickle('bow/bow'+str(i)+'_'+str(j)+'.pkl')
        corpus_r = corpus_r + bow

with open('corpus_r.pkl', 'wb') as fp:
    pickle.dump(corpus_r, fp)
    
df = pd.DataFrame()
nfile_lst = [57, 57, 56, 56, 56]
for i in range(1, 6):
    for j in range(nfile_lst[i-1]):
        _df = pd.read_pickle('doc_wordlist/agg'+str(i)+'_'+str(j)+'.pkl')
        _df = _df.iloc[:, :-3]
        df = df.append(_df)
df.to_pickle('df_bow.pkl')

df = pd.read_pickle('df_bow.pkl')
with open('corpus_r.pkl', 'rb') as fp:
    corpus_r = pickle.load(fp)
df['bow'] = corpus_r

df.to_pickle('df_bow_ret.pkl')