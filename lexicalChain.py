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
   
#@st.experimental_memo
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


def lexChain():
    with open('simple wikipedia.txt', "r", encoding="utf-8" ) as f:
        input_txt = f.read() 
        f.close()
        
    """
    Return the nouns of the entire text.
    """
    position = ['NN', 'NNS', 'NNP', 'NNPS']
    
    sentence = nltk.sent_tokenize(input_txt)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = [tokenizer.tokenize(w) for w in sentence]
    tagged =[pos_tag(tok) for tok in tokens]
    nouns = [word.lower() for i in range(len(tagged)) for word, pos in tagged[i] if pos in position ]
        
    relation = relation_list(nouns)
    lexical = create_lexical_chain(nouns, relation)
    final_chain = prune(lexical)
    """
    Print the lexical chain. 
    """   
    for i in range(len(final_chain)):
        print("Chain "+ str(i+1) + " : " + str(final_chain[i]))
    f=open('chain.txt','w',encoding='utf-8')
    for i in range(len(final_chain)):
        f.write("Chain "+ str(i+1) + " : " + str(final_chain[i]))
    # for i in range(len(final_chain)):
    #     final_chain[i] = str(final_chain[i])
    # le = preprocessing.LabelEncoder()
    # X = le.fit_transform(final_chain)
    # X = X.reshape(-1, 1)
    
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, df['Type'], test_size=0.33, random_state=50)
    # naive_bayesian_model = GaussianNB()
    # naive_bayesian_model.fit(X_train, y_train)
    # y_predict = naive_bayesian_model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_predict)
    # conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_predict)
    # precision,recall,fscore,non=precision_recall_fscore_support(y_test, y_predict,average='weighted')
    # return [accuracy,precision,recall,fscore]

lexChain()