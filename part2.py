import pandas as pd
from pyparsing import Word
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer
from nltk import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords
import spacy
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
import nltk
import streamlit as st
import json

nlp = spacy.load('en_core_web_sm')



@st.experimental_memo
def _get_nouns(text):
    target = []
    dict={}
    for lst in text:
        doc = nlp(lst)
        for token in doc:
            if len(token)>1:
                if token.tag_ in ['NN', 'NNP','NNS','NNPS']:
                    if token.text not in dict:
                        dict[token.text]=1
                    else:
                        dict[token.text]+=1
                        
    return dict
            # break
@st.experimental_memo
def bigram():
    df = pd.read_csv('data4.csv')
    dict={}
    dict=_get_nouns(df['Tokens'])
    #dict=sorted(dict.items(), key=lambda x: x[1], reverse=True)
    filtered_nouns=[]
    # print(dict[0])
    for i,key in enumerate(sorted(dict.items(), key=lambda x: x[1], reverse=True)):
        if i==50:
            break
        else:
            filtered_nouns.append(key[0])
    print(filtered_nouns,len(filtered_nouns))
    # bigrams= list(map(' '.join, nltk.bigrams(filtered_nouns)))
    lst2=[]
    for doc in df['Tokens']:
        
        bigrams=list(nltk.bigrams(doc.split()))
        #print(bigrams)
        lst=[]
        for gram in bigrams:
            if gram[0] in filtered_nouns or gram[1] in filtered_nouns:
                lst.append(gram)
            
        lst2.append(lst)
    # print(filtered_nouns)
    # print(lst2)    
    for i in range(len(lst2)):
        lst2[i] = str(lst2[i])
    le = preprocessing.LabelEncoder()
    df = pd.read_csv('data4.csv')
    X = le.fit_transform(lst2)
    X = X.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, df['Type'], test_size=0.33, random_state=5)

    # print(X_train)
    # print(X.shape)

    naive_bayesian_model = GaussianNB()
    naive_bayesian_model.fit(X_train, y_train)
    y_predict = naive_bayesian_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_predict)
    precision,recall,fscore,non=precision_recall_fscore_support(y_test, y_predict,average='weighted')
    return [accuracy,precision,recall,fscore]

