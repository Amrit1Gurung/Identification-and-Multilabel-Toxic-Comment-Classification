# -*- coding: utf-8 -*-
"""
Created on Fri May 13 02:20:31 2021

@author: Amrit Gurung
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 15 12:50:04 2021

@author: Amrit Gurung
"""


import re
import tqdm
from tqdm import tqdm
import pickle
import pandas as pd 
import numpy as np 
import streamlit as st
import nltk
from nltk.corpus import stopwords

# loading pickled vectorizer and models
with open('saved_models/tfidfVectorizer.pkl','rb') as file:
    tfidf = pickle.load(file)

with open('saved_models/model_toxic.pkl','rb') as file:
    model_toxic = pickle.load(file)

with open('saved_models/model_severe_toxic.pkl','rb') as file:
    model_severe_toxic = pickle.load(file)

with open('saved_models/model_obscene.pkl','rb') as file:
    model_obscene = pickle.load(file)

with open('saved_models/model_insult.pkl','rb') as file:
    model_insult = pickle.load(file)

with open('saved_models/model_threat.pkl','rb') as file:
    model_threat = pickle.load(file)

with open('saved_models/model_identity_hate.pkl','rb') as file:
    model_identity_hate = pickle.load(file)

# model lists
model_list = [model_toxic, model_severe_toxic, model_obscene, model_insult, model_threat, model_identity_hate]

stop_words = set(stopwords.words('english'))
# function to preprocess and remove stopwords
def preprocess(text):
    no_stops = []
    sent = text.lower()
    sent = re.sub(r"[^a-zA-Z]"," ",sent)
    sent = re.sub(r'\s+'," ",sent)
    #sent = str(text)
    for w in sent.split():
        if not w in stop_words:
            no_stops.append(w)
    return (" ".join(no_stops))

def Execute(x_test, option=True):
    category_list = ['toxic', 'severe_toxic', 'obscene', 'insult', 'threat', 'identity_hate']
    accuracies=[]
    selected_category = []
    if option:
        for model, category in tqdm(zip(model_list,category_list)):
            prediction = model.predict(x_test)
            if len(x_test) == 1:
                if option:
                    if prediction[0] == 1:
                        selected_category.append(category)
            else:
                pass
                #print('Test accuracy for {} is {}'.format(category,metrics.accuracy_score(y_test[category],prediction)))
                #accuracies.append(metrics.accuracy_score(y_test[category], prediction))
                #print('precision_recall_fscore_support_weighted', precision_recall_fscore_support(y_test[category], prediction, average='weighted'))
    #print('mean: ', sum(accuracies)/len(accuracies)) if option and len(accuracies)!=0 else None
    return selected_category

def predict():
    st.title("Toxic Comment Classification")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Toxic Comment/Text Classifier ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    test_str = st.text_area('Input comment/text tocheck for toxic categories','input text.....',height=200)
    test_string = preprocess(test_str)
    tfidf_test_string = tfidf.transform(list([test_string])).toarray()
    if st.button('Execute'):
        if not test_str:
            st.warning('Please input your text')
            st.stop()
        else:
            predicted_category = Execute(tfidf_test_string)
            if len(predicted_category)==0:
                st.success('The given comment/text is not toxic')
            else:
                st.success('Toxic Category: {}'.format(predicted_category))
    st.subheader('Model comparison')
    st.text('Comparison of varioous models')
    df = pd.read_csv('model comparison/model_comparison.csv')
    st.dataframe(df.style.highlight_max(axis=0))
    
    # for wordCloud viewing
    st.subheader('WordCloud')
    option = st.selectbox(
           'Select from the following toxic Category',
           ('Toxic', 'Severe Toxic', 'Insult','Obscene','Threat','Identity Hate'))
    if option == 'Toxic':
        st.image('plots/wordCloud/toxic.png',caption='Toxic Category wordCloud',width=400)
    elif option == 'Severe Toxic':
        st.image('plots/wordCloud/severe_toxic.png',caption='Severe Toxic wordCloud')
    elif option == 'Insult':
        st.image('plots/wordCloud/insult.png',caption='Insult wordCloud')
    elif option == 'Obscene':
        st.image('plots/wordCloud/obscene.png', caption='Obscene wordCloud')
    elif option == 'Threat':
        st.image('plots/wordCloud/threat.png',caption='Threat wordCloud')
    elif option == 'Identity Hate':
        st.image('plots/wordCloud/identity_hate.png',caption='Identity Hate wordCloud')
    else:
        pass
  

if __name__=='__main__':
    predict()
    










