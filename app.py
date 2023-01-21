import imp
import streamlit as st
import pandas as pd
import numpy as np 
from tfidf import tfIdf
from part2 import bigram
from lexicalChain import lexChain
from combineFeatures import combined
padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True
)
data={
    'Accuracy':0,
    'Precision':0,
    'Recall':0,
    'Fscore':0
}

st.markdown("<h1 style='text-align: center; color: green;'>Text Classification And Categorization</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: lightblue;'>Evaluation</h3>", unsafe_allow_html=True)
with st.sidebar:
    add_radio = st.radio(
        "Choose From Below:",
        ("Term Frequency/ Inverse Document Frequency (tf*idf)", "Topic Terms co-occurrence based",
         "Lexical Chains"," Mixed Features (Combining all above three)")
    )
    
if add_radio=="Term Frequency/ Inverse Document Frequency (tf*idf)":
        a,p,r,f=tfIdf()
        data['Accuracy']=a
        data['Precision']=p
        data['Recall']=r
        data['Fscore']=f
        df=pd.DataFrame(data,index=[0])
        st.table(df)
elif add_radio=="Topic Terms co-occurrence based":
        a,p,r,f=bigram()
        data['Accuracy']=a
        data['Precision']=p
        data['Recall']=r
        data['Fscore']=f
        df=pd.DataFrame(data,index=[0])
        st.table(df)
elif add_radio=="Lexical Chains":
        a,p,r,f=lexChain()
        data['Accuracy']=a
        data['Precision']=p
        data['Recall']=r
        data['Fscore']=f
        df=pd.DataFrame(data,index=[0])
        st.table(df)
elif add_radio==" Mixed Features (Combining all above three)":
        a,p,r,f=combined()
        data['Accuracy']=a
        data['Precision']=p
        data['Recall']=r
        data['Fscore']=f
        df=pd.DataFrame(data,index=[0])
        st.table(df)