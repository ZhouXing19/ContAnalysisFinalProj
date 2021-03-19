import pandas as pd
import pickle
from gensim.models import TfidfModel
import numpy as np
import matplotlib.pyplot as plt
import wordcloud
import warnings
warnings.filterwarnings("ignore")

# Read in the bow dataframe
df = pd.read_pickle('df_ind_bow.pkl')

# Read in the dictionary
with open('dictionary_r.pkl', 'rb') as fp:
    dictionary_r = pickle.load(fp)

df['FDATE'] = pd.to_datetime(df['FDATE'].astype(str))
corpus = df['bow'].values.tolist()
model = TfidfModel(corpus)

df['bow_dict'] = df['bow'].apply(dict)
word_ids = dictionary_r.keys()
term_freq = np.zeros((len(word_ids), 2), dtype = int)
term_freq[:, 0] = list(range(len(word_ids)))

for i in range(len(df)):
    words = df.iloc[i, -1]
    for word in words:
        term_freq[word][1]+= words[word]

df_tf = pd.DataFrame(term_freq)
df_tf.columns = ['word_id', 'term_freq']
df_tf['word'] = df_tf['word_id'].apply(lambda x: dictionary_r[x])
word_dict_all = df_tf[['word', 'term_freq']].set_index('word').to_dict()['term_freq']

# Plot overall wordcloud
wc = wordcloud.WordCloud(background_color="white", random_state=111111,max_words=500, width= 1600, height = 800, mode ='RGBA', scale=.5).generate_from_frequencies(word_dict_all)
plt.figure( figsize=(20,10), facecolor='k' )
plt.imshow(wc)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig('wordcloud_all.png', facecolor = 'k', bbox_inches = 'tight')

# Plot wordcloud for each sub-industry
df = df.dropna(subset = ['gind'])
df['gind_top'] = df['gind'].astype(int)//10000

def plot_industry(_df = df, ind = 40):
    ind_name = {40: 'Fin', 45: 'IT', 15: 'Material', 10: "Energy"}
    
    _df = _df[_df['gind_top'] == ind]
    
    _df['bow_tfidf'] = _df.iloc[:,-3].apply(lambda x: model[x])
    _df['bow_tfidf'] = _df['bow_tfidf'].apply(dict)
    
    term_freq = np.zeros((len(word_ids), 2))
    term_freq[:, 0] = list(range(len(word_ids)))
    
    for i in range(len(_df)):
        words = _df.iloc[i, -1]
        for word in words:
            term_freq[word][1]+= words[word]
            
    df_tf = pd.DataFrame(term_freq)
    df_tf.columns = ['word_id', 'term_freq']
    
    df_tf['word'] = df_tf['word_id'].apply(lambda x: dictionary_r[x])
    df_tf = df_tf.sort_values(by = 'term_freq', ascending = False).iloc[2:]
    df_tf = df_tf[df_tf['word']!='']
    
    word_dict = df_tf[['word', 'term_freq']].set_index('word').to_dict()['term_freq']
    
    wc = wordcloud.WordCloud(background_color="white", random_state=111111,max_words=500, width= 800, height = 800, mode ='RGBA', scale=.5).generate_from_frequencies(word_dict)
    plt.figure( figsize=(20,10), facecolor='k' )
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig('wordcloud_'+ind_name[ind]+'.png', facecolor = 'k', bbox_inches = 'tight')
    
    return df_tf

# Finance Industry
df_tf_fin = plot_industry(df, 40)

# IT Industry
df_tf_IT = plot_industry(df, 45)

# Energy Industry
df_tf_Eng = plot_industry(df, 10)

# Material Industry
df_tf_Mat = plot_industry(df, 15)

# Make plot of risks versus uncertainties
def get_word(x, id):
    try:
        return x[id]
    except:
        return 0
df['N_risk'] = df['bow_dict'].apply(lambda x: get_word(x, 970))
df['N_risks'] = df['bow_dict'].apply(lambda x: get_word(x, 1695))
df['N_uncertainty'] = df['bow_dict'].apply(lambda x: get_word(x, 2212))
df['N_uncertainties'] = df['bow_dict'].apply(lambda x: get_word(x, 2211))

df['year'] = pd.DatetimeIndex(df['FDATE']).year
risk_uncertainty = df[['year', 'N_risk', 'N_risks', 'N_uncertainty', 'N_uncertainties']].sort_values(by = 'year').groupby('year').sum()

# Read in Loughran-McDonald word lists
LM_neg = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name = 'Negative', header = None)
LM_pos = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name = 'Positive', header = None)
LM_uncertain = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name = 'Uncertainty', header = None)
LM_litigious = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name = 'Litigious', header = None)
LM_SM = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name = 'StrongModal', header = None)
LM_WM = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name = 'WeakModal', header = None)
LM_constraining = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx', sheet_name = 'Constraining', header = None)

# Get the ID of LM words in our dictionary
def get_id(x):
    try:
        return dictionary_r.token2id[x]
    except:
        return np.nan

def get_wordid(df):
    df[0] = df[0].apply(lambda x: x.lower())
    df['word_id'] = df[0].apply(get_id)
    df = df.dropna()
    return df

LM_neg = get_wordid(LM_neg)
LM_pos = get_wordid(LM_pos)
LM_uncertain = get_wordid(LM_uncertain)
LM_litigious = get_wordid(LM_litigious)
LM_SM = get_wordid(LM_SM)
LM_WM = get_wordid(LM_WM)
LM_constraining = get_wordid(LM_constraining)

# Create a vector of word frequency of LM word lists
def create_vect(x):
    vect = [0,0,0,0,0,0,0]
    neg = LM_neg['word_id'].values
    pos = LM_pos['word_id'].values
    uncertain = LM_uncertain['word_id'].values
    litigious = LM_litigious['word_id'].values
    SM = LM_SM['word_id'].values
    WM = LM_WM['word_id'].values
    constraining = LM_constraining['word_id'].values
    
    for word in neg:
        if word in x:
            vect[0] += x[word]
    
    for word in pos:
        if word in x:
            vect[1] += x[word]
    
    for word in uncertain:
        if word in x:
            vect[2] += x[word]
            
    for word in litigious:
        if word in x:
            vect[3] += x[word]
    
    for word in SM:
        if word in x:
            vect[4] += x[word]
            
    for word in WM:
        if word in x:
            vect[5] += x[word]
            
    for word in constraining:
        if word in x:
            vect[6] += x[word]
            
    return vect

df['vect'] = df['bow_dict'].apply(create_vect)

# Get the number of each category of words
df['N_LM_neg'] = df['vect'].apply(lambda x: x[0])
df['N_LM_pos'] = df['vect'].apply(lambda x: x[1])
df['N_LM_uncertain'] = df['vect'].apply(lambda x: x[2])
df['N_LM_litigious'] = df['vect'].apply(lambda x: x[3])
df['N_LM_SM'] = df['vect'].apply(lambda x: x[4])
df['N_LM_WM'] = df['vect'].apply(lambda x: x[5])
df['N_LM_constraining'] = df['vect'].apply(lambda x: x[6])

df_LM_factors = df[['FDATE', 'FName', 'N_LM_neg', 'N_LM_pos', 'N_LM_uncertain', 'N_LM_litigious', 'N_LM_SM', 'N_LM_WM', 'N_LM_constraining']]

df_LM_factors['year'] = pd.DatetimeIndex(df_LM_factors['FDATE']).year
df_LM_factors['month'] = pd.DatetimeIndex(df_LM_factors['FDATE']).month

# Use rolling average to create monthly LM factors
df_LM_month = df_LM_factors[['year', 'month', 'N_LM_neg', 'N_LM_pos', 'N_LM_uncertain', 'N_LM_litigious', 'N_LM_SM', 'N_LM_WM', 'N_LM_constraining']].groupby(['year', 'month']).mean()
df_LM_month.reset_index(inplace = True)
df_LM_month['date'] = pd.to_datetime((df_LM_month['year']*10000+df_LM_month['month']*100+1).astype(str))
df_LM_month['MA_neg'] = df_LM_month['N_LM_neg'].rolling(window = 12).mean()
df_LM_month['MA_pos'] = df_LM_month['N_LM_pos'].rolling(window = 12).mean()
df_LM_month['MA_uncertain'] = df_LM_month['N_LM_uncertain'].rolling(window = 12).mean()
df_LM_month['MA_litigious'] = df_LM_month['N_LM_litigious'].rolling(window = 12).mean()
df_LM_month['MA_SM'] = df_LM_month['N_LM_SM'].rolling(window = 12).mean()
df_LM_month['MA_WM'] = df_LM_month['N_LM_WM'].rolling(window = 12).mean()
df_LM_month['MA_constraining'] = df_LM_month['N_LM_constraining'].rolling(window = 12).mean()

# plot negative word factor
plt.plot(df_LM_month['date'], df_LM_month['MA_neg'])
plt.title('Negative Words')
plt.xlabel('Time')
plt.ylabel('Average Occurrence in 10-K MD&A')
print('e.g. abandon, abnormal, challenge')

# plot positive word factor
plt.plot(df_LM_month['date'], df_LM_month['MA_pos'])
plt.title('Positive Words')
plt.xlabel('Time')
plt.ylabel('Average Occurrence in 10-K MD&A')
print('e.g. able, achieve, advancement')

# plot uncertainty word factor
plt.plot(df_LM_month['date'], df_LM_month['MA_uncertain'])
plt.title('Uncertainty Words')
plt.xlabel('Time')
plt.ylabel('Average Occurrence in 10-K MD&A')
print('e.g. almost, ambiguity, risk, volatile')

# plot litigious word factor
plt.plot(df_LM_month['date'], df_LM_month['MA_litigious'])
plt.title('Litigious Words')
plt.xlabel('Time')
plt.ylabel('Average Occurrence in 10-K MD&A')
print('e.g. lawsuit, jury, legal')

# plot strong modal word factor
plt.plot(df_LM_month['date'], df_LM_month['MA_SM'])
plt.title('Strong Modal Words')
plt.xlabel('Time')
plt.ylabel('Average Occurrence in 10-K MD&A')
print('e.g. always, best, clearly')

# plot weak modal word factor
plt.plot(df_LM_month['date'], df_LM_month['MA_WM'])
plt.title('Weak Modal Words')
plt.xlabel('Time')
plt.ylabel('Average Occurrence in 10-K MD&A')
print('e.g. almost, appear, could, depend')

# plot contraining word factor
plt.plot(df_LM_month['date'], df_LM_month['MA_constraining'])
plt.title('Constraining Words')
plt.xlabel('Time')
plt.ylabel('Average Occurrence in 10-K MD&A')
print('e.g.: abide, bound, commit')
