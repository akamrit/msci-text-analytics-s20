# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 18:21:06 2020

@author: Amrit
"""

#C:/Users/Amrit Kaur/Desktop/assn1/Assignment_1
import sys
import re
from random import randrange
import csv
from doctest import OutputChecker
import math
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
positive.strip('\n')
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

#After this path, the declared Corpus is worked upon
#Now the corpus has been defined



def tokenize(data):
    data=data.strip() # remove trailing and leading spaces
    data=re.sub("[^A-Za-z0-9 .\n -]","",data)# remove special characters
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

def data_split(data_sw,data,train,val,lab):
    #This is to split the data in 3 sets. train and val contains the percentage of the data to be split...for ex train =0.80
    len_train=math.floor(len(data)*train)
    global lable
    len_val=math.ceil(len(data)*val)
    for i in range(0,len_train):
        #making training set
        index=randrange(len(data))
        training_set_without_sw.append(data[index])
        training_set_sw.append(data_sw[index])
        data.pop(index)
        data_sw.pop(index)
        lab_train.append(lab[index])
        lab.pop(index)
    for i in range(0,len_val):
        #making Validation Set
        index=randrange(len(data))
        validation_set_without_sw.append(data[index])
        validation_set_sw.append(data_sw[index])
        data.pop(index)
        data_sw.pop(index)
        lab_val.append(lab[index])
        lab.pop(index)
    for i in data:
        test_set_without_sw.append(i)
    for i in data_sw:
        test_set_sw.append(i)
    for i in lable:
        lab_test.append(i)

def write_file(data,name):
    with open(name+'.csv','w',newline='') as output:
        for i in data:
            csv.writer(output).writerow(i)


corpus=corpus.replace('\n',' \n ')#adding trailing and leading space in corpus for easy processing 
corpus=corpus.lower()
corpus_sw=corpus
#Working on making two sets i.e with and without stop words.
corpus_without_sw=remove_sw(corpus)
tokenized_data_sw=tokenize(corpus_sw)
tokenized_data_without_sw=tokenize(corpus_without_sw)

write_file(tokenized_data_sw,'out')
write_file(tokenized_data_without_sw,'out_ns')



# Tokenized files havee been generated. Now splitting the data in 3 sets i.e train,val,test


len_train=80
len_train=float(len_train)/100
len_val=10
len_val=float(len_val)/100
training_set_without_sw=[]
validation_set_without_sw=[]
test_set_without_sw=[]
training_set_sw=[]
validation_set_sw=[]
test_set_sw=[]
lab_train=[]
lab_val=[]
lab_test=[]
print((lable))
data_split(tokenized_data_sw,tokenized_data_without_sw,len_train,len_val,lable)
write_file(training_set_without_sw,'train_ns')
write_file(training_set_sw,'train')
write_file(validation_set_without_sw,'val_ns')
write_file(validation_set_sw,'val')
write_file(test_set_without_sw,'test_ns')
write_file(test_set_sw,'test')
write_file(lab_train,'lab_train')
write_file(lab_val,'lab_val')
write_file(lab_test,'lab_test')
    
print('Corpus has been split and files have been generated.')
input('Thanks for Using the program. Exiting!')
