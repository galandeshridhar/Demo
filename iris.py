# -*- coding: utf-8 -*-
"""
Created on Wed Dec 02 05:15:03 2020
@author: Shridhar Galande
"""

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

st.write('''
         # Simple Iris Flower Prediction App
         This app Predicts Iris Flower Type!
         ''')

st.sidebar.header("User Input Data")

def User_Input_features():
    Sepal_Length=st.sidebar.slider("Sepal Length",4.3,7.9,5.4)
    Sepal_Width= st.sidebar.slider("Sepal Width",2.0,4.4,3.4)
    Petal_Length=st.sidebar.slider("Petal Length",1.0,6.9,1.4)
    Petal_Width= st.sidebar.slider("Petal Width",0.1,2.5,0.2)
    
    data={"Sepal_Length":Sepal_Length,
          "Sepal_Width":Sepal_Width,
          "Petal_Length":Petal_Length,
          "Petal_Width":Petal_Width
        }
   
    Features=pd.DataFrame(data,index=[0])
    return Features

df=User_Input_features()
st.subheader("User Input Data")
st.write(df)

# load the iris dataset
iris=datasets.load_iris()
X=iris.data
Y=iris.target

# Build random forest classifier
Rd=RandomForestClassifier()
Rd.fit(X,Y)

Prediction=Rd.predict(df)
Prediction_prob=Rd.predict_proba(df)

st.subheader("Types of Flowers with thier corresponding  Index Values")
st.write(iris.target_names)

st.header("1. Random Forest Classifier")
st.subheader("Predicted Flower: ")
st.write(iris.target_names[Prediction_prob.argmax()])

st.subheader("Prediction Probability Depending On Flower Index Value: ")
st.write(Prediction_prob)


#Build Decision tree classifier
DT=DecisionTreeClassifier()
DT.fit(X,Y)

Prediction=DT.predict(df)
Prediction_prob=DT.predict_proba(df)

st.header("2. Decision Tree Classifier")
st.subheader("Predicted Flower: ")
st.write(iris.target_names[Prediction_prob.argmax()])

st.subheader("Prediction Probability Depending On Flower Index Value: ")
st.write(Prediction_prob)


# # Build Logistic regression
# LR=LogisticRegression()
# LR.fit(X,Y)

# Prediction=LR.predict(df)
# Prediction_prob=LR.predict_proba(df)

# st.header("3. Logistic Regression")
# st.subheader("Predicted Flower: ")
# st.write(iris.target_names[Prediction_prob.argmax()])

# st.subheader("Prediction Probability Depending On Flower Index Value: ")
# st.write(Prediction_prob)


# Build SVM
clf = SVC(gamma='auto')
clf.fit(X,Y)

Prediction=clf.predict(df)
# Prediction_prob=clf.predict_proba(df)

st.header("4. Support Vector Machine")
st.subheader("Predicted Flower: ")
st.write(iris.target_names[Prediction][0])

# Build SVM
NB = MultinomialNB()
NB.fit(X,Y)

Prediction=NB.predict(df)
Prediction_prob=NB.predict_proba(df)
# Prediction_prob=clf.predict_proba(df)

st.header("5. Naive Bayes")
st.subheader("Predicted Flower: ")
st.write(iris.target_names[Prediction_prob.argmax()])

st.subheader("Prediction Probability Depending On Flower Index Value: ")
st.write(Prediction_prob)


page_bg_img = '''
<style>
body {
background-image: url("https://images.all-free-download.com/images/graphiclarge/beautiful_white_flower_vector_background_541646.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
			header:before {
				content:'Created by @Shridhar Galande@'; 
				visibility: visible;
			    top:10px;    
			    position: fixed;
  				right: 45%;
				#background-color: orange;
				#color:red;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 