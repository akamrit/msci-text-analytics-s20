# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 18:59:19 2020

@author: Amrit
"""

from gensim.models import  Word2Vec as wv
vector_model=wv.load('./data/w2v.model') # loading the file
import pprint

with open(input("Enter the test file path for evaluation"),'r') as test:
    file=test.read()
    file=file.split('\n')
    result={}
    
    for i in file:
        if i =="":
            continue
        i.strip()
        #finding similarities
        words=vector_model.most_similar(i,topn=20)
        temp=[]
        #now generating result dictionary
        for j in words:
            temp.append(j[0])
        result[i]=temp
pprint.pprint(result)

