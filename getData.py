import codecs
from matplotlib.pyplot import text
import requests
from cleanText import Preprocess
import os
from bs4 import BeautifulSoup
import nltk.stem as ns
from nltk.stem import WordNetLemmatizer
import nltk
import json
import re
import spacy
from nltk.corpus import stopwords
import csv
import pandas as pd
from nltk.stem.porter import *
nlp = spacy.load('en_core_web_sm')
stemmer = PorterStemmer()
# nltk.download('stopwords')
all_stopwords = stopwords.words('english')
header = ['document','tokens','bi-grams']
dict={}
course=[]
non_course=[]
data=[]
folderpath='course-cotrain-data/fulltext/'
filename='fulltext.csv'
def preprocessor(text):
    text = re.sub('http\S+\s*', ' ', text)  # remove URLs
    text = re.sub('RT|cc', ' ', text)  # remove RT and cc
    text = re.sub('#\S+', '', text)  # remove hashtags
    text = re.sub('@\S+', '  ', text)  # remove mentions
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)  # remove punctuations
    text = re.sub(r'[^\x00-\x7f]',r' ', text) 
    text = re.sub('\s+', ' ', text)  # remove extra whitespace
    text = text.lower()  # convert to lowercase
    translator = str.maketrans("", "", "'!@#$%^&*()_=-\|][:';:,<.>/?`~")
    x = text.translate(translator)
    tokens=nltk.word_tokenize(x)
    tokens=[word for word in tokens if not word in stopwords.words()]
    print(tokens)
    return text

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def _get_target_words(text):
    target = []
    sent = " ".join(text)
    doc = nlp(sent)
    for token in doc:
        if token.tag_ in ['NN', 'NNP','NNS','NNPS']:
            target.append(token.text)
    return target
def remove_tags(text, postags=['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV']):
    filtered = []
    str_text = nlp(" ".join(text))
    for token in str_text:
        if token.pos_ in postags:
            filtered.append(token.text)
    return filtered

for k,folder in enumerate(os.listdir(folderpath)):
    for i,file in enumerate(os.listdir(folderpath+folder)):
        f = open(folderpath+folder+'/'+file, "r",errors='ignore')
        html=f.read()
        soup = BeautifulSoup(html, "lxml")
        cleaned_text=cleanhtml(str(soup))
        pr=Preprocess()
        text=pr.preprocess(cleaned_text)
        text = ' '.join([str(elem) for elem in text])
        print(text)
        data.append((file,text))
        dict={'Document':data[i][0],
                'Tokens':data[i][1],
                'Type':k}
        file_exists = os.path.isfile('data4.csv')
        with open('data4.csv','a',encoding='utf-8',newline='') as f:
                wr=csv.DictWriter(f,fieldnames=dict.keys())
                if not file_exists:
                    wr.writeheader()
                wr.writerow(dict)
