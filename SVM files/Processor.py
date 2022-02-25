import nltk
import ssl 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
import numpy as np
import pandas as pd

def makeTextColumn(input_df):

    columns = ['title', 'body', 'author', 'dateline']
    input_df[columns] = input_df[columns].replace(np.nan, '')
    input_df["processedData"] = (input_df["title"] + " " +  input_df["body"] + 
    " " +  input_df["author"] + " " +  input_df['dateline'])
    input_df.drop(columns=["title", "body", 'author', 'dateline'], inplace=True)

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    r = re.compile(r"[\w']+")
    porter = PorterStemmer()
    stop_words = list(stopwords.words('english'))
    stop_words.append("reuter")
    stop_words.append("reuters")

    for ind, val in enumerate(input_df["processedData"]):
        tokenized_list = word_tokenize(val)
        tokenized_list = [words.lower() for words in tokenized_list]
        tokenized_list = [words for words in tokenized_list if words not in stop_words]
        tokenized_list = list(filter(r.match, tokenized_list))
        tokenized_list = [porter.stem(words) for words in tokenized_list]
        input_df.loc[ind, "processedData"] = " ".join(tokenized_list)

def categorizeEarn(input_df):
    for ind, val in enumerate(input_df["topics"]):
        input_df.loc[ind, "topics"] = 1 if "earn" in val else 0

def processDF(input_df):
    makeTextColumn(input_df)
    categorizeEarn(input_df)
    return input_df
