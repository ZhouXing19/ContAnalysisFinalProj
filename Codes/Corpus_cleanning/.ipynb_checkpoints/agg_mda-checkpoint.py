import glob
import pandas as pd
files = glob.glob('Corpus_mda/*')

files.sort()

df_agg1 = pd.DataFrame()
for i, file in enumerate(files[0:2000]):
#     print(i)
    df_agg1 = df_agg1.append(pd.read_pickle(file))

df_agg1.to_pickle('mda_agg/mda_agg1.pkl')

df_agg2 = pd.DataFrame()
for i, file in enumerate(files[2000:4000]):
#     print(i)
    df_agg2 = df_agg2.append(pd.read_pickle(file))

df_agg2.to_pickle('mda_agg/mda_agg2.pkl')

df_agg3 = pd.DataFrame()
for i, file in enumerate(files[4000:6000]):
#     print(i)
    df_agg3 = df_agg3.append(pd.read_pickle(file))

df_agg3.to_pickle('mda_agg/mda_agg3.pkl')

del df_agg3

df_agg4 = pd.DataFrame()
for i, file in enumerate(files[6000:8000]):
#     print(i)
    df_agg4 = df_agg4.append(pd.read_pickle(file))

df_agg4.to_pickle('mda_agg/mda_agg4.pkl')

df_agg5 = pd.DataFrame()
for i, file in enumerate(files[8000:]):
#     print(i)
    df_agg5 = df_agg5.append(pd.read_pickle(file))

df_agg5.to_pickle('mda_agg/mda_agg5.pkl')