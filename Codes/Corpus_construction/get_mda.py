import nest_asyncio
import unicodedata
nest_asyncio.apply()

from secedgar.filings import Filing, FilingType
import datetime as dt
import bs4 
import re
import glob
import pandas as pd
import os
import shutil


def get_mda(id):

    def normalize_text(text):
        """Normalize Text
        """
        text = unicodedata.normalize("NFKD", text)  # Normalize
        text = '\n'.join(text.splitlines())  # Unicode break lines

        # Convert to upper
        text = text.upper()  # Convert to upper

        # Take care of breaklines & whitespaces combinations due to beautifulsoup parsing
        text = re.sub(r'[ ]+\n', '\n', text)
        text = re.sub(r'\n[ ]+', '\n', text)
        text = re.sub(r'\n+', '\n', text)

        # To find MDA section, reformat item headers
        text = text.replace('\n.\n', '.\n')  # Move Period to beginning

        text = text.replace('\nI\nTEM', '\nITEM')
        text = text.replace('\nITEM\n', '\nITEM ')
        text = text.replace('\nITEM  ', '\nITEM ')

        text = text.replace(':\n', '.\n')

        # Math symbols for clearer looks
        text = text.replace('$\n', '$')
        text = text.replace('\n%', '%')

        # Reformat
        text = text.replace('\n', '\n\n')  # Reformat by additional breakline

        return text

    def find_mda_from_text(text, start=0):
        """Find MDA (Management Discussion and Analysis) section from normalized text
        Args:
            text (str)s
        """
        debug = False

        mda = ""
        end = 0

        # Define start & end signal for parsing
        item7_begins = [
            '\nITEM 7.', '\nITEM 7 –', '\nITEM 7:', '\nITEM 7 ', '\nITEM 7\n'
        ]
        item7_ends = ['\nITEM 7A']
        if start != 0:
            item7_ends.append('\nITEM 7')  # Case: ITEM 7A does not exist
        item8_begins = ['\nITEM 8']
        """
        Parsing code section
        """
        text = text[start:]

        # Get begin
        for item7 in item7_begins:
            begin = text.find(item7)
            if debug:
                print(item7, begin)
            if begin != -1:
                break

        if begin != -1:  # Begin found
            for item7A in item7_ends:
                end = text.find(item7A, begin + 1)
                if debug:
                    print(item7A, end)
                if end != -1:
                    break

            if end == -1:  # ITEM 7A does not exist
                for item8 in item8_begins:
                    end = text.find(item8, begin + 1)
                    if debug:
                        print(item8, end)
                    if end != -1:
                        break

            # Get MDA
            if end > begin:
                mda = text[begin:end].strip()
            else:
                end = 0

        return mda, end

#     df_names = pd.read_csv('master/stocknames_form10k.csv')
    
#     df_names.drop_duplicates(subset=['gvkey'], inplace = True)
    
    df_names = pd.read_pickle('df_company.pkl')
    
    
    gvkeys = df_names['gvkey'].values[id*14:(id+1)*14]
    names = df_names['CoName'].values[id*14:(id+1)*14]

    for j in range(len(gvkeys)):
        my_filings = Filing(cik_lookup = names[j], filing_type = FilingType.FILING_10K, 
                            start_date = dt.datetime(2010,1,1), end_date = dt.datetime(2020,12,31))

        try:
            my_filings.save('Corpus_10k')
        except:
            continue

        company = names[j]

        files = glob.glob('Corpus_10k/'+company+'/10-k/*')
        files.sort()

        df = pd.DataFrame()
        for i in files:
            print(i)
            try:
                with open(i) as f:
                    content = f.read()
            except:
                continue
            try:
                soup = bs4.BeautifulSoup(content, "html.parser")
            except:
                continue
            text = soup.get_text("\n")
            text = normalize_text(text)
            mda, end = find_mda_from_text(text)
            if mda and len(mda.encode('utf-8')) < 1000:
                mda, _ = find_mda_from_text(text, start=end)
            if len(mda.encode('utf-8')) < 1000:
                continue
            df = df.append(pd.DataFrame({'company': [company], 'filename': [i], 'mda': [mda]}))

        df.to_pickle('Corpus_mda/'+str(gvkeys[j])+'_mda.pkl')
        shutil.rmtree('Corpus_10k/'+company)
        
    return 0