#!/bin/env python
import pandas as pd
import lucem_illud
import gensim

def norm_text(idx):
    df = pd.read_csv('agg/mda_agg4.csv')
    
    if idx < 55:
        df = df.iloc[idx*126:126*(idx+1),:]
    else:
        df = df.iloc[idx*126:,:]
    
    df['mda'] = df['mda'].apply(lambda x: x.replace('\n\n', ' '))

    df['tokenized_mda'] = df['mda'].apply(lambda x: lucem_illud.word_tokenize(x))

    df['normalized_tokens'] = df['tokenized_mda'].apply(lambda x: lucem_illud.normalizeTokens(x))
    
    df.to_pickle('doc_wordlist/agg4_'+str(idx)+'.pkl')
    
    return 0