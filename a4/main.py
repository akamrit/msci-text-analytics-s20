# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 19:47:21 2020

@author: Amrit
"""

import sys
import os
import csv
from pprint import pprint
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
import pandas 
import pickle

import tensorflow as tf
import keras
import numpy as np
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense,Input,LSTM,Embedding,Dropout,Flatten
from tensorflow.keras import regularizers
vocab_size=0


def convert_file_to_list(file_name):
    with open(data_file+'\\'+file_name) as file:
        file=csv.reader(file,delimiter='\n')
        list_form=[]
        for i in file:
            for j in i:
                j.strip('][')
                j.replace('.','')
                list_form.append(j)
        return list_form

def convert_lab_to_list(lab_name):
    with open(data_file+'\\'+lab_name) as file:
        file=file.read()
        lable=[]
        for i in file:
            if i=='\n':
                continue
            if i=='0':
                i=0
            elif i=='1':
                i=1
            lable.append(i)
        return lable
    
def encode_and_pad(data,vocabulary):
    #encoding the dataset into integer format to be fed into training modeland then padding it
    encoded_data=[]
    for sentence in data:
        temp=[]
        sentence='<sos>,'+sentence+',<eos>'
        for word in sentence.split(','):
            en_id=vocabulary.get(word)
            if en_id==None:
                continue
            else:
                temp.append(en_id)
        encoded_data.append(temp)
    #In amazon corpus, 27 is the max length
    encoded_data= tf.keras.preprocessing.sequence.pad_sequences(encoded_data,maxlen=27,padding='post')
    return encoded_data

def embedding_matrix_and_vocabulary(train_x):
     #loading word 2 vec model and generating vocabulary 
    vector_model=Word2Vec.load('a3/data/w2v.model')
    print(vector_model)

    #vocab size
    global vocab_size
    vocab_size=len(vector_model.wv.vocab)
    print(vocab_size)
    
    '''
    #save model
    emd='w2v.txt'
    model.wv.save_word2vec_format(emd, binary=False)
    #print(emd)

    emb_index={}
    file1=open(os.path.join('','w2v.txt'),encoding="utf-8")
    line=file1.readlines()[1:]
    file1.close()
    for i in line:
        w=i.split()
        wd=w[0]
        v=np.asarray(w[1:],dtype='float32')
        emb_index[wd]=v
    file1.close()
    '''

    #Generating Vocabulary of training data:
    t=Tokenizer()
    t.fit_on_texts(train_x)
    #encoded_train_x=t.texts_to_sequences(train_x)
    #print(t.word_index)
    vocabulary=t.word_index
    print(len(vocabulary))

    vocab_size+=2 #for sos and eos

    #Adding eos and sos in the vocabulary



    #print(encoded_train_x)
    #print(encoded_test_x)
    #encoded_test_x=t.texts_to_sequences(test_x)

    #encoded_val_x=t.texts_to_sequences(val_x)


    #word_index={k:(v+3) for k,v in word_index.items()}
    #word_index["<PAD>"]=0
    #word_index["<START>"]=1
    #word_index["<UNK>"]=2
    #word_index["<UNUSED>"]=3

    #reverse_word_index=dict([(value,key) for (key,value) in word_index.items()])

    #train_x=tf.keras.preprocessing.sequence.pad_sequences


    embedding_matrix=np.zeros((vocab_size+1,300))

    for i,j in vocabulary.items():
        try:
            
            embedding_matrix[j+2]=vector_model[i]
            
            vocabulary[i]=j+2
        except KeyError:
            continue
    
    vocabulary['<sos>']=1
    vocabulary['<eos>']=2
        
    
    return embedding_matrix ,vocabulary

def generate_model(embedding_matrix,activation_function):
    #define model
    print('Building Model')
    model=Sequential()

    embedding_layer = Embedding(
            input_dim=vocab_size+1,
            input_length=27,
            output_dim=300,
            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
            trainable=False)
    model.add(embedding_layer)

    model.add(Dropout(0.2))
    model.add(Dense(1,activation=activation_function,kernel_regularizer=regularizers.l2(0.01)))
    model.add(Flatten())
    model.add(Dense(1,activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    model.fit(encoded_train_x,lab_train,batch_size=200,epochs=25,verbose=False,validation_data=(encoded_val_x,lab_val))
    
    result=model.evaluate(encoded_test_x,lab_test)
    print("Accuracy: %.2f%%" % (result[1]*100))
    return model

data_file=input("Input folder name for the given datasets")
train_x=convert_file_to_list('train.csv')
test_x=convert_file_to_list('test.csv')
val_x=convert_file_to_list('val.csv')

lab_train=convert_lab_to_list('lab_train.csv')
lab_test=convert_lab_to_list('lab_test.csv')
lab_val=convert_lab_to_list('lab_val.csv')

#Preprocessing the training data and generating 
embedding_matrix,vocabulary=embedding_matrix_and_vocabulary(train_x)


encoded_train_x=encode_and_pad(train_x,vocabulary)
#print(embedding_matrix)
#print(encoded_train_x)
#print(type(train_file_encoded))

encoded_test_x=encode_and_pad(test_x,vocabulary)
#print(encoded_test_x)
#print(type(encoded))
encoded_val_x=encode_and_pad(val_x,vocabulary)
#print(encoded_val_x)

nn_relu=generate_model(embedding_matrix,'relu')

nn_sigmoid=generate_model(embedding_matrix,'sigmoid')

nn_tanh=generate_model(embedding_matrix,'tanh')

nn_relu.save('./data/nn_relu.model')
nn_sigmoid.save('./data/nn_sigmoid.model')
nn_tanh.save('./data/nn_tanh.model')

with open('./data/vocabulary.pkl','wb') as pic_file:
        pickle.dump(vocabulary,pic_file)
