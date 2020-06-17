# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 19:16:05 2020

@author: Amrit
"""

import pickle
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

file_path=input('Please Input evaluation file path:')
#file_path='C:\\Users\\Amrit Kaur\\Desktop\\pos.txt'
with open(file_path) as file:
    file=file.read()
file=file.split('\n')
print(file)
classifier=input('Input classifier Name:')
with open(classifier+'.pkl', 'rb') as pic:
    vector=pickle.load(pic)
    nb_model=pickle.load(pic)

#transforming the file based on given vector vocabulary 
file_trans=vector.transform(file)
pred=nb_model.predict(file_trans)
print(pred)
