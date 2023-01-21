from bs4 import BeautifulSoup
import re
import os
import nltk
nltk.download('punkt')
import matplotlib.pyplot as plt
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import gensim
import string
import nltk.data
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from spellchecker import SpellChecker
import spacy
import csv
import pandas as pd
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")
data=[]
dict={}


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

folderpath='course-cotrain-data/inlinks/'

import gensim
import string

spell = SpellChecker()

def correct_spelling(text):
    corrected_text = list()
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        next_word = word
        if word in misspelled_words:
            next_word = spell.correction(word)
        corrected_text.append(next_word)
    
    return " ".join(corrected_text)

stop_words = stopwords.words('english')
def remove_stopwords(tokenized_sentences):
    lst=[]
    for token in tokenized_sentences:
        if token not in stop_words:
           lst.append(token)
    return lst

# Lemmatize all words
wordnet_lemmatizer = WordNetLemmatizer()
def lemmatize_words(tokenized_sentences):
    lst=[]
    for token in tokenized_sentences:
        tk=wordnet_lemmatizer.lemmatize(token)
        lst.append(tk)
    return lst

snowball_stemmer = SnowballStemmer('english')
def stem_words(tokenized_sentences):
    lst=[]
    for token in tokenized_sentences:
        tk=snowball_stemmer.stem(token)
        lst.append(tk)
    return lst
# Uses gensim to process the sentences
def sentence_to_words(sentences):
    for sentence in sentences:
        sentence_tokenized = gensim.utils.simple_preprocess(sentence,
                                                            deacc=True,
                                                            min_len=2,
                                                            max_len=15)
        
        # Make sure we don't yield empty arrays
        if len(sentence_tokenized) > 0:
            yield sentence_tokenized

# Process the sentences manually
def sentence_to_words_from_scratch(sentences):
    for sentence in sentences:
        sentence_tokenized = [token.lower() for token in 
               word_tokenize(sentence.translate(str.maketrans('','',string.punctuation)))]
        
        # Make sure we don't yield empty arrays
        if len(sentence_tokenized) > 0:
            yield sentence_tokenized
# Helper method for generating n-grams
def extract_ngrams_sentences(sentences, num):
    all_grams = []
    for sentence in sentences:
        n_grams = ngrams(sentence, num)
        all_grams += [ ' '.join(grams) for grams in n_grams]
    return all_grams

# Splits text up by newline and period
def split_by_newline_and_period(pages):
    sentences = []
    for page in pages:
        sentences += re.split('\n', page)
    return sentences
def condense_newline(text):
    return '\n'.join([p for p in re.split('\n|\r', text) if len(p) > 0])
def parse_html(html_text):
        soup = html_text
        TAGS = ['p','span','br','h1','h2','h3','h4','h5','h6','strong','em','q','blockquote','li','ul','ol','dl','dt','dd','mark','ins','del','sup','sub','small','i','b']
        return ' '.join([condense_newline(tag.text) for tag in soup.findAll(TAGS)])
    
    
if __name__=='__main__':
    course_flag=0
    count=0
    for k,folder in enumerate(os.listdir(folderpath)):
        for i,file in enumerate(os.listdir(folderpath+folder)):
                count+=1 
                if folder=='course':
                    course_flag=1
                elif folder=='non-course':
                    course_flag=0
                print(folderpath+folder+'/'+file,course_flag)
                f = open(folderpath+folder+'/'+file, "r",errors='ignore')
                html=f.read()
                soup = BeautifulSoup(html, "lxml")
                parsed_text=parse_html(soup)
                sentences=re.findall('[\w]+', parsed_text)
                sentences=" ".join(str(token) for token in sentences)
                sentences=sentences.lower()
                sentences=re.sub('[\t\n\r\f ]+', ' ', sentences)
                sentences=re.sub(r'http\S+', '', sentences)
                print(sentences)
                sentences=correct_spelling(sentences)
                sentences=sentences.translate(str.maketrans('', '', string.punctuation))
                shortword = re.compile(r'\W*\b\w{1,2}\b')
                sentences=shortword.sub('', sentences)
                sentences=nltk.word_tokenize(sentences)
                sentences = remove_stopwords(sentences)
                sentences = lemmatize_words(sentences)
                sentences = stem_words(sentences)
                text=" ".join(str(token) for token in sentences)
                data.append((file,text))
                dict={'Document':data[i][0],
                        'Tokens':data[i][1],
                        'Type':course_flag}
                file_exists = os.path.isfile('data6.csv')
                with open('data6.csv','a',encoding='utf-8',newline='') as f:
                        wr=csv.DictWriter(f,fieldnames=dict.keys())
                        if not file_exists:
                            wr.writeheader()
                        wr.writerow(dict)
    