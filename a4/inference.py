# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 21:06:26 2020

@author: Amrit
"""

from gensim.models import  Word2Vec as wv
import csv
import sklearn
import tensorflow as tf
import numpy as np
import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.initializers import Constant
from keras.regularizers import l2
from keras.layers.embeddings import Embedding
import pickle
import re
vocab_size=0

def encode_and_pad(data,vocabulary):
    #encoding the dataset into integer format to be fed into training modeland then padding it
    encoded_data=[]
    
    for sentence in data:
        temp=[]
        sentence='<sos> '+sentence+' <eos>'
        for word in sentence.split(' '):
            en_id=vocabulary.get(word)
            if en_id==None:
                continue
            else:
                temp.append(en_id)
        encoded_data.append(temp)
    #In amazon corpus, 27 is the max length
    encoded_data= pad_sequences(encoded_data,maxlen=27,padding='post')
    return encoded_data

file_path=input('Enter test file path')
model_name=input('Model name(relu,sigmoid,tanh)')

with open(file_path) as file:
    file=file.read()
    file=re.sub("[^A-Za-z0-9 \n -]","",file)
    file=file.lower()
file=file.split('\n')
print(file)

if(model_name=='relu'):
    
    model=tf.keras.models.load_model('data\\nn_relu.model')
    
if(model_name=='sigmoid'):
    
    model=tf.keras.models.load_model('data\\nn_sigmoid.model')
    
if(model_name=='tanh'):
    
    model=tf.keras.models.load_model('data\\nn_tanh.model')
   

with open('data/vocabulary.pkl', 'rb') as pic:
    vocabulary=pickle.load(pic)
    
file_encoded=encode_and_pad(file,vocabulary)
print(file_encoded)

ynew = model.predict(file_encoded)
# show the inputs and predicted outputs
for i in range(len(file_encoded)):
    print("X=%s, Predicted=%s" % (file[i], ynew[i]))
    
print(vocabulary)

