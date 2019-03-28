import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

np.random.seed(500)

from nltk.stem import WordNetLemmatizer
tokens = word_tokenize(phrases[0])
tokens = [w.lower() for w in tokens]
print ("Lemmatisation \n")
wordnet_lemmatizer = WordNetLemmatizer()
lstemmed = [wordnet_lemmatizer.lemmatize(word,pos='v') for word in tokens]
print("Lemmatisation : \n",lstemmed)


#Prepare Train and Test Data sets

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['text_final'],df['label'],test_size=0.3)

#Encoding
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

#Word Vectorization
def Word_Vector(Train_X,Test_X)
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(df['text_final'])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    
    return (Tfidf_vect, Train_X_Tfidf,Test_X_Tfidf)

word_vector= Word_Vector(Train_X,Test_X)
print(word_vector[0].vocabulary_)
print(word_vector[2])


#The SVM — Support Vector Machine

#The Naive Bayes Classifier Algorithm
