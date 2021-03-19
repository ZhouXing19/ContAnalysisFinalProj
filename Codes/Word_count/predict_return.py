###############################################
# Use all word count to predict stock returns 
###############################################

import pandas as pd
import pickle

hb = pickle.load(open('dictionary_r.pkl', 'rb'))
hb.filter_extremes(no_below=10, no_above=0.5)

word_ids = hb.keys()

df_main = pickle.load(open('df_bow_ret.pkl', 'rb'))

df_main['FDATE'] = pd.to_datetime(df_main['FDATE'].astype(str))

import datetime as dt
df_train = df_main[df_main['FDATE']<dt.datetime(2019,1,1)]

df_test = df_main[df_main['FDATE']>=dt.datetime(2019,1,1)]

df_train = df_train[['bow', 'ma_ret']]
# df_main = df_main[['bow' ,'RET']]

df_train.columns = ['words', 'ret']

df_train['words'] = df_train['words'].apply(lambda x: [y[0] for y in x])

df_train['ret'] = pd.to_numeric(df_train['ret'], errors = 'coerce')

df_main = df_train

import numpy as np
fv_arr = np.zeros((len(word_ids), 2), dtype=int)
for i in range(len(df_main)):
    words = df_main.iloc[i,0]
    ret = df_main.iloc[i,1]
    for word in words:
        fv_arr[word][0] += 1
        if ret > 0:
            fv_arr[word][1] += 1

FValue = pd.DataFrame(fv_arr)

FValue.columns = ['Denom', 'Nom']

FValue['index'] = FValue.index

FValue['word'] = FValue['index'].apply(lambda x: hb[x])

FValue = FValue[FValue['Denom']!=0]

FValue['FValue'] = FValue['Nom']/FValue['Denom']

pd.set_option('max_row', 200)
neg_df = FValue[FValue['Denom']>2000].sort_values(by='FValue').head(200)[['word', 'FValue']]

pos_df = FValue[FValue['Denom']>2000].sort_values(by='FValue', ascending = False).head(100)[['word', 'FValue']]

pos_words = pos_df['word'].values.tolist()
neg_words = neg_df['word'].values.tolist()

df_main = pickle.load(open('df_bow_ret.pkl', 'rb'))
df_main['FDATE'] = pd.to_datetime(df_main['FDATE'].astype(str))
df_train = df_main[df_main['FDATE']<dt.datetime(2019,1,1)]

df_train['words'] = df_train['bow'].apply(dict)

pos_id = [hb.token2id[i] for i in pos_words]

neg_id = [hb.token2id[i] for i in neg_words]

all_id = neg_id + pos_id

def get_vec(x):
    vec = [0]*len(all_id)
    for i, word in enumerate(all_id):
        if word in x:
            vec[i] += x[word]
    return vec

df_train['vect'] = df_train['words'].apply(get_vec)

df_test['words'] = df_test['bow'].apply(dict)
df_test['vect'] = df_test['words'].apply(get_vec)

df_train['label'] = np.where(df_train['ma_ret']>=0, 1, 0)
df_test['label'] = np.where(df_test['ma_ret']>=0, 1, 0)

from sklearn.linear_model import LogisticRegression
logistic_ret= LogisticRegression(penalty='l2')
logistic_ret.fit(np.stack(df_train['vect'], axis=0), df_train['label'])

print('Logistic Regression:')
print('training: ', logistic_ret.score(np.stack(df_train['vect'], axis=0), df_train['label']))
print('testing: ', logistic_ret.score(np.stack(df_test['vect'], axis=0), df_test['label']))

import sklearn
from sklearn.ensemble import RandomForestClassifier
rf_ret = sklearn.ensemble.RandomForestClassifier()
rf_ret.fit(np.stack(df_train['vect'], axis =0), df_train['label'])

print('Random Forest')
print('training: ', rf_ret.score(np.stack(df_train['vect'], axis=0), df_train['label']))
print('testing: ', rf_ret.score(np.stack(df_test['vect'], axis=0), df_test['label']))

from sklearn.neural_network import MLPClassifier
nn_ret = sklearn.neural_network.MLPClassifier()
nn_ret.fit(np.stack(df_train['vect'], axis =0), df_train['label'])

print('Multi-layer perceptron')
print('training: ', nn_ret.score(np.stack(df_train['vect'], axis=0), df_train['label']))
print('testing: ', nn_ret.score(np.stack(df_test['vect'], axis=0), df_test['label']))

###############################################
# Use financial dictionary word count to predict stock returns 
###############################################
import warnings
warnings.filterwarnings("ignore")

df_main = pickle.load(open('df_bow_ret.pkl', 'rb'))

df_main['FDATE'] = pd.to_datetime(df_main['FDATE'].astype(str))

LM_neg = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name = 'Negative', header = None)

LM_neg.columns = ['neg']

LM_neg['neg'] = LM_neg['neg'].apply(lambda x: x.lower())

LM_pos = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name = 'Positive', header = None)

LM_pos.columns = ['pos']

LM_pos['pos'] = LM_pos['pos'].apply(lambda x: x.lower())

LM_words = LM_neg['neg'].values.tolist() + LM_pos['pos'].values.tolist()

LM_lst = []
word_lst = []
for i in LM_words:
    try:
        LM_lst.append(hb.token2id[i])
        word_lst.append(i)
    except:
        continue

import datetime as dt
df_train = df_main[df_main['FDATE']<dt.datetime(2019,1,1)]

df_test = df_main[df_main['FDATE']>=dt.datetime(2019,1,1)]

def get_vec(x):
    vec = [0]*len(LM_lst)
    for i, word in enumerate(LM_lst):
        if word in x:
            vec[i] += x[word]
    return vec

df_train['words'] = df_train['bow'].apply(dict)

df_train['vect'] = df_train['words'].apply(get_vec)

df_test['words'] = df_test['bow'].apply(dict)
df_test['vect'] = df_test['words'].apply(get_vec)

df_train['label'] = np.where(df_train['ma_ret']>=0, 1, 0)

df_test['label'] = np.where(df_test['ma_ret']>=0, 1, 0)

logistic_ret= LogisticRegression(penalty='l2')
logistic_ret.fit(np.stack(df_train['vect'], axis=0), df_train['label'])
print('logistic regression')
print('training: ', logistic_ret.score(np.stack(df_train['vect'], axis=0), df_train['label']))
print('testing: ', logistic_ret.score(np.stack(df_test['vect'], axis=0), df_test['label']))

import sklearn
from sklearn.ensemble import RandomForestClassifier
rf_ret = sklearn.ensemble.RandomForestClassifier()
rf_ret.fit(np.stack(df_train['vect'], axis =0), df_train['label'])
print('random forest')
print('training: ', rf_ret.score(np.stack(df_train['vect'], axis=0), df_train['label']))
print('testing: ', rf_ret.score(np.stack(df_test['vect'], axis=0), df_test['label']))

from sklearn.neural_network import MLPClassifier
nn_ret = sklearn.neural_network.MLPClassifier()
nn_ret.fit(np.stack(df_train['vect'], axis =0), df_train['label'])
print('multi-layer perceptron')
print('training: ', nn_ret.score(np.stack(df_train['vect'], axis=0), df_train['label']))
print('testing: ', nn_ret.score(np.stack(df_test['vect'], axis=0), df_test['label']))

###############################################
# Use all word tf-idf to predict stock returns 
###############################################
import pandas as pd
import pickle

hb = pickle.load(open('dictionary_r.pkl', 'rb'))
hb.filter_extremes(no_below=10, no_above=0.5)

word_ids = hb.keys()

df_main = pickle.load(open('df_bow_ret.pkl', 'rb'))

from gensim.models import TfidfModel

corpus = df_main['bow'].values.tolist()

model = TfidfModel(corpus)

df_main['bow_tfidf'] = df_main['bow'].apply(lambda x: model[x])

df_main['FDATE'] = pd.to_datetime(df_main['FDATE'].astype(str))

import datetime as dt
df_train = df_main[df_main['FDATE']<dt.datetime(2019,1,1)]
df_test = df_main[df_main['FDATE']>=dt.datetime(2019,1,1)]

df_FValue = df_train[['bow', 'ma_ret']]
df_FValue.columns = ['words', 'ret']

df_FValue['words'] = df_FValue['words'].apply(lambda x: [y[0] for y in x])
df_FValue['ret'] = pd.to_numeric(df_FValue['ret'], errors = 'coerce')

import numpy as np
fv_arr = np.zeros((len(word_ids), 2), dtype=int)
for i in range(len(df_FValue)):
    words = df_FValue.iloc[i,0]
    ret = df_FValue.iloc[i,1]
    for word in words:
        fv_arr[word][0] += 1
        if ret > 0:
            fv_arr[word][1] += 1


FValue = pd.DataFrame(fv_arr)
FValue.columns = ['Denom', 'Nom']
FValue['index'] = FValue.index
FValue['word'] = FValue['index'].apply(lambda x: hb[x])
FValue = FValue[FValue['Denom']!=0]
FValue['FValue'] = FValue['Nom']/FValue['Denom']

pd.set_option('max_row', 200)
neg_df = FValue[FValue['Denom']>2000].sort_values(by='FValue').head(200)[['word', 'FValue']]
pos_df = FValue[FValue['Denom']>2000].sort_values(by='FValue', ascending = False).head(100)[['word', 'FValue']]

pos_words = pos_df['word'].values.tolist()
neg_words = neg_df['word'].values.tolist()

pos_id = [hb.token2id[i] for i in pos_words]
neg_id = [hb.token2id[i] for i in neg_words]

all_id = neg_id + pos_id

def get_vec(x):
    vec = [0]*len(all_id)
    for i, word in enumerate(all_id):
        if word in x:
            vec[i] += x[word]
    return vec

df_train['words'] = df_train['bow_tfidf'].apply(dict)
df_train['vect'] = df_train['words'].apply(get_vec)

df_test['words'] = df_test['bow_tfidf'].apply(dict)
df_test['vect'] = df_test['words'].apply(get_vec)

df_train['label'] = np.where(df_train['ma_ret']>=0, 1, 0)
df_test['label'] = np.where(df_test['ma_ret']>=0, 1, 0)

from sklearn.linear_model import LogisticRegression
logistic_ret= LogisticRegression(penalty='l2')
logistic_ret.fit(np.stack(df_train['vect'], axis=0), df_train['label'])

print('Logistic Regression:')
print('training: ', logistic_ret.score(np.stack(df_train['vect'], axis=0), df_train['label']))
print('testing: ', logistic_ret.score(np.stack(df_test['vect'], axis=0), df_test['label']))

import sklearn
from sklearn.ensemble import RandomForestClassifier
rf_ret = sklearn.ensemble.RandomForestClassifier()
rf_ret.fit(np.stack(df_train['vect'], axis =0), df_train['label'])

print('Random Forest')
print('training: ', rf_ret.score(np.stack(df_train['vect'], axis=0), df_train['label']))
print('testing: ', rf_ret.score(np.stack(df_test['vect'], axis=0), df_test['label']))

from sklearn.neural_network import MLPClassifier
nn_ret = sklearn.neural_network.MLPClassifier()
nn_ret.fit(np.stack(df_train['vect'], axis =0), df_train['label'])

print('Multi-layer perceptron')
print('training: ', nn_ret.score(np.stack(df_train['vect'], axis=0), df_train['label']))
print('testing: ', nn_ret.score(np.stack(df_test['vect'], axis=0), df_test['label']))

###############################################
# Use LM dictionary tf-idf to predict stock returns 
###############################################
import warnings
warnings.filterwarnings("ignore")

df_main = pickle.load(open('df_bow_ret.pkl', 'rb'))
df_main['FDATE'] = pd.to_datetime(df_main['FDATE'].astype(str))
from gensim.models import TfidfModel
corpus = df_main['bow'].values.tolist()

model = TfidfModel(corpus)
df_main['bow_tfidf'] = df_main['bow'].apply(lambda x: model[x])


LM_neg = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name = 'Negative', header = None)
LM_neg.columns = ['neg']
LM_neg['neg'] = LM_neg['neg'].apply(lambda x: x.lower())

LM_pos = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name = 'Positive', header = None)
LM_pos.columns = ['pos']
LM_pos['pos'] = LM_pos['pos'].apply(lambda x: x.lower())

LM_words = LM_neg['neg'].values.tolist() + LM_pos['pos'].values.tolist()

LM_lst = []
word_lst = []
for i in LM_words:
    try:
        LM_lst.append(hb.token2id[i])
        word_lst.append(i)
    except:
        continue

import datetime as dt
df_train = df_main[df_main['FDATE']<dt.datetime(2019,1,1)]
df_test = df_main[df_main['FDATE']>=dt.datetime(2019,1,1)]

def get_vec(x):
    vec = [0]*len(LM_lst)
    for i, word in enumerate(LM_lst):
        if word in x:
            vec[i] += x[word]
    return vec

df_train['words'] = df_train['bow_tfidf'].apply(dict)
df_train['vect'] = df_train['words'].apply(get_vec)

df_test['words'] = df_test['bow_tfidf'].apply(dict)
df_test['vect'] = df_test['words'].apply(get_vec)

df_train['label'] = np.where(df_train['ma_ret']>=0, 1, 0)

df_test['label'] = np.where(df_test['ma_ret']>=0, 1, 0)

logistic_ret= LogisticRegression(penalty='l2')
logistic_ret.fit(np.stack(df_train['vect'], axis=0), df_train['label'])
print('logistic regression')
print('training: ', logistic_ret.score(np.stack(df_train['vect'], axis=0), df_train['label']))
print('testing: ', logistic_ret.score(np.stack(df_test['vect'], axis=0), df_test['label']))

import sklearn
from sklearn.ensemble import RandomForestClassifier
rf_ret = sklearn.ensemble.RandomForestClassifier()
rf_ret.fit(np.stack(df_train['vect'], axis =0), df_train['label'])
print('random forest')
print('training: ', rf_ret.score(np.stack(df_train['vect'], axis=0), df_train['label']))
print('testing: ', rf_ret.score(np.stack(df_test['vect'], axis=0), df_test['label']))

from sklearn.neural_network import MLPClassifier
nn_ret = sklearn.neural_network.MLPClassifier()
nn_ret.fit(np.stack(df_train['vect'], axis =0), df_train['label'])
print('multi-layer perceptron')
print('training: ', nn_ret.score(np.stack(df_train['vect'], axis=0), df_train['label']))
print('testing: ', nn_ret.score(np.stack(df_test['vect'], axis=0), df_test['label']))


###############################################
# Use Topic Attention to predict stock returns 
###############################################
import pandas as pd
import numpy as np

topic_df = pd.read_pickle('topic_change_10_topics.pickle')

def get_vect(x):
    lst = [0]*10
    y = dict(x)
    for i in range(10):
        if i in y:
            lst[i]+=y[i]
    return lst

topic_df['vect'] = topic_df['topics'].apply(get_vect)

df = pd.read_pickle('df_ind_bow.pkl')

df['vect'] = topic_df['vect']

df['RET'] = pd.to_numeric(df['RET'], errors = 'coerce')

df['label'] = np.where(df['ma_ret']>=0, 1, 0)

df['FDATE'] = pd.to_datetime(df['FDATE'].astype(str))

import datetime as dt
df_train = df[df['FDATE']<dt.datetime(2019,1,1)]
df_test = df[df['FDATE']>=dt.datetime(2019,1,1)]

from sklearn.linear_model import LogisticRegression
logistic_ret= LogisticRegression(penalty='l2')
logistic_ret.fit(np.stack(df_train['vect'], axis=0), df_train['label'])

print('Logistic Regression:')
print('training: ', logistic_ret.score(np.stack(df_train['vect'], axis=0), df_train['label']))
print('testing: ', logistic_ret.score(np.stack(df_test['vect'], axis=0), df_test['label']))

import sklearn
from sklearn.ensemble import RandomForestClassifier
rf_ret = sklearn.ensemble.RandomForestClassifier()
rf_ret.fit(np.stack(df_train['vect'], axis =0), df_train['label'])

print('Random Forest')
print('training: ', rf_ret.score(np.stack(df_train['vect'], axis=0), df_train['label']))
print('testing: ', rf_ret.score(np.stack(df_test['vect'], axis=0), df_test['label']))
