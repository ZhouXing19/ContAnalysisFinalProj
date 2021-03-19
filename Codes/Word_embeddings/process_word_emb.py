#All these packages need to be installed from pip
import gensim#For word2vec, etc
import argparse
import math
import numpy as np #For arrays
import pandas as pd #Gives us DataFrames
pd.options.mode.chained_assignment = None

import os #For looking through files
import os.path #For managing file paths

import io

import matplotlib.pyplot as plt #For graphics
import seaborn #Makes the graphics look nicer
import sklearn.metrics.pairwise #For cosine similarity
import sklearn.manifold #For T-SNE
import sklearn.decomposition #For PCA


import pickle


class GetEmbeddings:
    def __init__(self,
                 sector,
                 textDatadfPath="./Data/MDA_2010_2020/textDatadf",
                 industryDataPath="./Data/MDA_2010_2020/master_industry.csv",
                 industryDescriptionPath="./Data/MDA_2010_2020/GICS_map_2018.xlsx",
                 filterDictPath="./Data/MDA_2010_2020/Zhou/dictionary_r_unfiltered.pkl"
    ):
        self.sector = sector

        assert self.sector in ['Information Technology', 'Financials', 'Energy', 'Materials']

        self.textDatadfPath = textDatadfPath
        self.industryDataPath = industryDataPath
        self.industryDescriptionPath = industryDescriptionPath
        self.filterDictPath = filterDictPath
        self.load_industry_code()
        self.sector_code = self.IndCodeTable[self.sector]
        self.filter_dict = None

        self.load_text_df()


    def load_text_df(self):
        textDatadfFiles = os.listdir(self.textDatadfPath)
        self.textDatadf = pd.DataFrame()


        for idx, file in enumerate(textDatadfFiles):
            filepath = os.path.join(self.textDatadfPath, file)
            this_df = pickle.load(open(filepath, 'rb'))
            #filtered_df = this_df[:5]
            filtered_df = this_df.loc[this_df['gind'] == self.sector_code]
            filtered_df['normalized_tokens'] = filtered_df['normalized_tokens'].apply(lambda x: [x])
            self.textDatadf = self.textDatadf.append(filtered_df)
            if idx % 5 == 0:
                print(f'====finished : [{idx}] files====')
        print('----finished load_text_df----')
        return

    def load_fname_gind(self):
        # 其实用不上你
        industryData = pd.read_csv(self.industryDataPath)
        self.FName_gind = industryData[['FName', 'gind']]
        self.FName_gind.gind = self.FName_gind.gind.apply(self.ChangeCode).astype('Int64')

    def load_industry_code(self):
        industryDescData = pd.read_excel(self.industryDescriptionPath)
        indData = industryDescData[industryDescData.columns[0:2]].dropna()
        indData.columns = ['codes', 'indName']
        self.CodeIndTable = indData.set_index('codes').to_dict()['indName']
        self.IndCodeTable = {v: k for k, v in self.CodeIndTable.items()}
        return

    def load_filter_dict(self):
        self.filter_dict = pickle.load(open(self.filterDictPath, 'rb'))
        ori_len = len(self.filter_dict)
        to_keeps = ['systematic', 'unsystematic', 'political', 'regulatory', 'financial', 'interest', 'rate', 'country',
                    'social', 'environmental', 'operational', 'management', 'legal', 'competition', 'economic', 'compliance',
                    'security','fraud', 'operational', 'operation', 'competition', 'risk', 'uncertainty', 'uncertainties', 'risks',
                    'personnel', 'salary', 'wage', 'pandemic', 'covid', 'covid-19', 'epidemic', 'health']

        self.filter_dict.filter_extremes(no_below=20, no_above=0.6, keep_tokens=to_keeps)
        updated_len = len(self.filter_dict)
        print(f"---- removed [{ori_len - updated_len}] words ----")

    def ChangeCode(self, val):
        if math.isnan(val):
            return val
        return int(str(val)[:2])

    def get_embeddings(self, save_wv = False):
        self.senReleasesW2V = gensim.models.word2vec.Word2Vec(self.textDatadf['normalized_tokens'].sum())
        if save_wv:
            self.save_wv()
        self.saveWordVecPairInFile(self.senReleasesW2V.wv)
        return

    def save_wv(self):
        assert self.senReleasesW2V != None
        pklPath = f'./Data/MDA_2010_2020/{self.sector}_wv.pkl'
        if os.path.exists(pklPath):
            os.remove(pklPath)
        with open(pklPath, 'wb') as handle:
            pickle.dump(self.senReleasesW2V.wv, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def saveWordVecPairInFile(self, wv, filtered=True):
        print(f'----Start writing files for [{self.sector}] ----')
        out_v_path = f'tsvs/{self.sector}_vectors.tsv'
        out_m_path = f'tsvs/{self.sector}_metadata.tsv'

        if os.path.exists(out_v_path):
            os.remove(out_v_path)

        if os.path.exists(out_m_path):
            os.remove(out_m_path)

        out_v = io.open(out_v_path, 'w', encoding='utf-8')
        out_m = io.open(out_m_path, 'w', encoding='utf-8')

        vocabulary = list(wv.vocab.keys())
        print(f'----Preview vocab for [{self.sector}] : {vocabulary[:5]}----')


        words = []
        if filtered:
            self.load_filter_dict()
            assert self.filter_dict != None
            words = set(self.filter_dict.values())

        for index, word in enumerate(vocabulary):
            if filtered and word in words:
                vec = wv[word]
                out_v.write('\t'.join([str(x) for x in vec]) + "\n")
                out_m.write(word + "\n")
        out_v.close()
        out_m.close()

        print(f'----Finished writing files for [{self.sector}] ----')





# textDatadfPath = "./Data/MDA_2010_2020/textDatadf"
# textDatadfFiles = os.listdir(textDatadfPath)
# newtextDatadf = pd.DataFrame()
# for file in textDatadfFiles:
#     filepath = os.path.join(textDatadfPath, file)
#     curDf = pickle.load(open(filepath, 'rb'))
#     newtextDatadf = newtextDatadf.append(curDf)

# demo_df = newtextDatadf.head()
# demo_df_cp = demo_df[:]


# Load Industry data
# industryDataPath = "./Data/MDA_2010_2020/master_industry.csv"
# industryDescriptionPath = "./Data/MDA_2010_2020/GICS_map_2018.xlsx"
#
# industryData = pd.read_csv(industryDataPath)
# FName_gind = industryData[['FName', 'gind']]
# FName_gind.gind = FName_gind.gind.apply(ChangeCode).astype('Int64')
#
# industryDescData = pd.read_excel(industryDescriptionPath)
# indData = industryDescData[industryDescData.columns[0:2]].dropna()
# indData.columns = ['codes', 'indName']
# CodeIndTable = indData.set_index('codes').to_dict()['indName']
# IndCodeTable = {v: k for k, v in CodeIndTable.items()}
# indCodes = set(CodeIndTable.keys())



# desiredIndustries = ['Information Technology', 'Financials', 'Energy', 'Materials']
# desiredCodes = set([IndCodeTable[industry] for industry in desiredIndustries])




# demo_df['normalized_tokens'] = demo_df['normalized_tokens'].apply(lambda x: [x])
# demo_list = demo_df['normalized_tokens'].sum()
# senReleasesW2V = gensim.models.word2vec.Word2Vec(demo_list)
# senReleasesW2V.wv.vocab.keys()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sector', default='Information Technology',
                        help='sector selected')

    opt = parser.parse_args()
    # opt.sector
    getEmbedding = GetEmbeddings(opt.sector)
    getEmbedding.get_embeddings(save_wv=True)

    # this_sec = "Information Technology"
    # this_ge = GetEmbeddings(this_sec)
    # this_ge.get_embeddings(save_wv=True)
    #
    # this_wv = this_ge.senReleasesW2V.wv
    #
    #
    # def normalize(vector):
    #     normalized_vector = vector / np.linalg.norm(vector)
    #     return normalized_vector
    #
    #
    # def dimension(model, positives, negatives):
    #     diff = sum([normalize(model[x]) for x in positives]) - sum([normalize(model[y]) for y in negatives])
    #     return diff
    #
    #
    # Risk_Uncertainty = dimension(this_wv, ['risk', 'risks', 'risky', 'risking', 'risk, '], ['uncertainty', 'uncertainties', 'uncertain'])
    # Key_Words = ['political', 'regulatory', 'financial', 'interest', 'rate', 'country',\
    #              'social', 'environmental', 'operational', 'management', 'legal', \
    #              'competition', 'economic', 'compliance', 'security','fraud', \
    #              'operational', 'operation', 'competition', ]
    #
    #
    # def makeDF(model, word_list):
    #     RU = []
    #     for word in word_list:
    #         RU.append(
    #             sklearn.metrics.pairwise.cosine_similarity(this_wv[word].reshape(1, -1), Risk_Uncertainty.reshape(1, -1))[
    #                 0][0])
    #     df = pd.DataFrame({'Risk_Uncertainty': RU}, index=word_list)
    #     return df