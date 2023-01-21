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
import spacy
import nltk
from sklearn import preprocessing
stemmer = SnowballStemmer("english")
all_stopwords = stopwords.words('english')



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
    # print(filtered_nouns,len(filtered_nouns))
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
    # for i in range(len(lst2)):
    #     lst2[i] = str(lst2[i])
        
    return lst2
    # le = preprocessing.LabelEncoder()
    # df = pd.read_csv('data4.csv')
    # X = le.fit_transform(lst2)
    # X = X.reshape(-1, 1)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, df['Type'], test_size=0.33, random_state=5)

    # # print(X_train)
    # # print(X.shape)

    # naive_bayesian_model = GaussianNB()
    # naive_bayesian_model.fit(X_train, y_train)
    # y_predict = naive_bayesian_model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_predict)
    # conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_predict)
    # precision,recall,fscore,non=precision_recall_fscore_support(y_test, y_predict,average='weighted')
    # return [accuracy,precision,recall,fscore]
@st.experimental_memo
def combined():
    # dt=[]
    df = pd.read_csv('data4.csv')
    tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words='english',tokenizer=word_tokenize, binary=True, max_features=100)
    # string=[]
    lst=[]
    lst2=[]
    bigram_list=bigram()
    for tup_doc in bigram_list:
        lst=[' '.join(tup) for tup in tup_doc] 
        lst2.append(lst)
    
    for i in range(len(lst2)):
        lst2[i] = str(" ".join(lst2[i]))
    print(lst2)
    df2=pd.DataFrame(lst2)
    # print(type(df2))
    terms = tfidfvectorizer.fit_transform(lst2)
    X_train, X_test, y_train, y_test = train_test_split(terms, df['Type'],test_size=0.33, random_state=9)
    naive_bayesian_model = GaussianNB()
    naive_bayesian_model.fit(X_train.todense(), y_train)
    y_predict = naive_bayesian_model.predict(X_test.todense())
    accuracy = accuracy_score(y_test, y_predict)
    # conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_predict,labels=[0,1])
    precision,recall,fscore,non=precision_recall_fscore_support(y_test, y_predict,beta=1.0, labels=None, pos_label=1, average='weighted', warn_for=('precision', 'recall', 'f-score'), sample_weight=None, zero_division='warn')
    print(accuracy,precision,recall,fscore)
    return [accuracy,precision,recall,fscore]
