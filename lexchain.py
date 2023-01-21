import codecs
from cvxpy import length
from matplotlib.pyplot import text
import requests
from cleanText import Preprocess
import os
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
import nltk
from sklearn.model_selection import train_test_split
import string
from heapq import nlargest
from nltk.tag import pos_tag
from string import punctuation
from inspect import getsourcefile
from collections import defaultdict
from nltk.tokenize import word_tokenize
from os.path import abspath, join, dirname
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import stanfordnlp

stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
doc.sentences[0].print_dependencies()
# nlp = spacy.load('en_core_web_sm')
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
    return tokens

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



    # print(text)



def relation_list(nouns):

    relation_list = defaultdict(list)
    
    for k in range (len(nouns)):   
        relation = []
        for syn in wordnet.synsets(nouns[k], pos = wordnet.NOUN):
            for l in syn.lemmas():
                relation.append(l.name())
                if l.antonyms():
                    relation.append(l.antonyms()[0].name())
            for l in syn.hyponyms():
                if l.hyponyms():
                    relation.append(l.hyponyms()[0].name().split('.')[0])
            for l in syn.hypernyms():
                if l.hypernyms():
                    relation.append(l.hypernyms()[0].name().split('.')[0])
        relation_list[nouns[k]].append(relation)
    return relation_list
    
def create_lexical_chain(nouns, relation_list):
    lexical = []
    threshold = 0.5
    for noun in nouns:
        flag = 0
        for j in range(len(lexical)):
            if flag == 0:
                for key in list(lexical[j]):
                    if key == noun and flag == 0:
                        lexical[j][noun] +=1
                        flag = 1
                    elif key in relation_list[noun][0] and flag == 0:
                        syns1 = wordnet.synsets(key, pos = wordnet.NOUN)
                        syns2 = wordnet.synsets(noun, pos = wordnet.NOUN)
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
                    elif noun in relation_list[key][0] and flag == 0:
                        syns1 = wordnet.synsets(key, pos = wordnet.NOUN)
                        syns2 = wordnet.synsets(noun, pos = wordnet.NOUN)
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
        if flag == 0: 
            dic_nuevo = {}
            dic_nuevo[noun] = 1
            lexical.append(dic_nuevo)
            flag = 1
    return lexical
   
def prune(lexical):
    final_chain = []
    while lexical:
        result = lexical.pop()
        if len(result.keys()) == 1:
            for value in result.values():
                if value != 1: 
                    final_chain.append(result)
        else:
            final_chain.append(result)
    return final_chain


def lexChain(input_txt):
    
    # df = pd.read_csv('data4.csv')
    # input_txt=df['Tokens']

    position = ['NN', 'NNS', 'NNP', 'NNPS']
    final_chain=[]
    for i in range(len(input_txt)):
        sentence = nltk.sent_tokenize(input_txt[i])
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = [tokenizer.tokenize(w) for w in sentence]
        tagged =[pos_tag(tok) for tok in tokens]
        print(tagged)
        nouns = [word.lower() for i in range(len(tagged)) for word, pos in tagged[i] if pos in position ]
            
        relation = relation_list(nouns)
        lexical = create_lexical_chain(nouns, relation)
        final_chain.append(prune(lexical))
    
    """
    Print the lexical chain. 
    """   
    for i in range(len(final_chain)):
        print("Chain "+ str(i+1) + " : " + str(final_chain[i]))
    #for i in range(len(final_chain)):
    #     final_chain[i] = str(final_chain[i])
        
        
        
with open('simple wikipedia.txt','r',errors='ignore') as f:
    text=f.read()
    # print(length(text))
    pr=Preprocess()
    text=pr.preprocess(text)
    text = ' '.join([str(elem) for elem in text])
    print(text[5])
    #lexChain(text)