# Calculate word F Values
import pandas as pd
import pickle
import numpy as np

hb = pickle.load(open('dictionary_r.pkl', 'rb'))
hb.filter_extremes(no_below=10, no_above=0.5)

word_ids = hb.keys()

# Get the words and the return
df_main = pickle.load(open('df_bow_ret.pkl', 'rb'))
df_main = df_main[['bow', 'ma_ret']]
df_main.columns = ['words', 'ret']
df_main['words'] = df_main['words'].apply(lambda x: [y[0] for y in x])
df_main['ret'] = pd.to_numeric(df_main['ret'], errors = 'coerce')

# Calculate F Value
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

# Plot the negative words
stopwords = ['policies', 'covering', 'discretionary', 'moves', 'survey', 'southeastern']
neg_df = FValue[FValue['Denom']>2000].sort_values(by='FValue').head(200)[['word', 'FValue']]
neg_df.at[2228, 'word'] = 'pressure'
neg_df[['FValue']] = 0.5 - neg_df['FValue']
word_dict_neg = neg_df[~neg_df.word.isin(stopwords)].set_index('word').to_dict()['FValue'] #.sort_values(by='Denom')

wc = wordcloud.WordCloud(background_color="black", random_state=111111,max_words=500, width= 1600, height = 800, mode ='RGBA', scale=.5, stopwords = stopwords).generate_from_frequencies(word_dict_neg)
plt.figure( figsize=(20,10), facecolor='k' )
plt.imshow(wc)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig('wordcloud_neg.png', facecolor = 'k', bbox_inches = 'tight')

# Plot the positive words
pos_df = FValue[FValue['Denom']>2000].sort_values(by='FValue', ascending = False).head(200)[['word', 'FValue']]
pos_df.at[2707, 'word'] = 'firm'
word_dict_pos = pos_df.set_index('word').to_dict()['FValue']

wc = wordcloud.WordCloud(background_color="white", random_state=111111,max_words=500, width= 1600, height = 800, mode ='RGBA', scale=.5, stopwords = stopwords).generate_from_frequencies(word_dict_pos)
plt.figure( figsize=(20,10), facecolor='k' )
plt.imshow(wc)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig('wordcloud_pos.png', facecolor = 'k', bbox_inches = 'tight')
