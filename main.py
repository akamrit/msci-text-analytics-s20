# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:05:00 2020

@author: Amrit
"""

import sys
import re
from random import randrange
import csv
from doctest import OutputChecker
import math
from gensim.models import  Word2Vec as wv
#define stopwords
stopwords=['i', "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

def set_path(data_pathe,name):
    #Initializing the data path and corpus text file. This is the starting Block
    global positive 
    flag=0
    with open(data_pathe+'\\'+name,'r') as corpus:
        positive=corpus.read()
        #print(corpus)
data_path=input('Please enter the data Path :') 
set_path(data_path,'pos.txt')
positive.strip(' ')
positive.strip('\\n')
corpus=positive

lable=[]
positive=positive.split('\n')
for i in positive:
    if i==' ' or i=='\n':
        continue
    lable.append('1')

set_path(data_path,'neg.txt')
positive.strip(' ')
positive.strip('\n')
corpus= corpus + '\n'+ positive
positive=positive.split('\n')
for i in positive:
    if i==' ' or i=='\n':
        continue
    lable.append('0')



def tokenize(data):
    data=data.strip() # remove trailing and leading spaces
    data=re.sub("[^A-Za-z0-9 \n -]","",data)# remove special characters
    data=data.splitlines()# split at line
    result=[]
    for i in data:
        #split word in each line
        temp=i.strip().split()
        result.append(temp)
    return result
   
        
def remove_sw(data):
    data=data.lower()
    for i in stopwords:
        data = re.sub("[^A-Za-z0-9]"+i+"[^A-Za-z0-9]", " ", data)
    return data



corpus=corpus.replace('\n',' \n ')#adding trailing and leading space in corpus for easy processing 
corpus=corpus.lower()
corpus_sw=corpus
#Working on making two sets i.e with and without stop words.
corpus_without_sw=remove_sw(corpus)
tokenized_data_sw=tokenize(corpus_sw)
tokenized_data_without_sw=tokenize(corpus_without_sw)


print("Training model")
vector_model=wv(tokenized_data_sw,min_count=1,window=5,workers=4,size=300)
vector_model.save('./data/w2v.model')
model = wv.load('./data/w2v.model')
print(vector_model)

print("positive words \n" , vector_model.wv.most_similar(['good'],topn=20))
print("negative words \n ", vector_model.wv.most_similar(['bad'],topn=20))