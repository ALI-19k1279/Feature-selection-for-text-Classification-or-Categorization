#import count vectorize and tfidf vectorise
import pandas as pd
from pyparsing import Word
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer
from nltk import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import streamlit as st
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
all_stopwords = stopwords.words('english')

@st.experimental_memo
def tfIdf():
    dt=[]
    df = pd.read_csv('data4.csv')
    tfidfvectorizer = TfidfVectorizer(tokenizer=word_tokenize,ngram_range=(1,1) ,binary=True, max_features=100)
    #terms = tfidfvectorizer.fit_transform(df['Tokens'].values.astype(str))
    terms = tfidfvectorizer.fit_transform(df['Tokens'])
    X_train, X_test, y_train, y_test = train_test_split(terms, df['Type'],test_size=0.33,random_state=9)
    naive_bayesian_model = GaussianNB()
    naive_bayesian_model.fit(X_train.todense(), y_train)
    y_predict = naive_bayesian_model.predict(X_test.todense())
    accuracy = accuracy_score(y_test, y_predict)
    precision,recall,fscore,non=precision_recall_fscore_support(y_test, y_predict,beta=1.0, average='weighted')
    print(accuracy,precision,recall,fscore)
    return [accuracy,precision,recall,fscore]

tfIdf()