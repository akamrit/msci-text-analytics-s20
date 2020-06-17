# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 19:11:58 2020

@author: Amrit
"""

import csv
import sklearn
import ast
import pickle
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
data_file=input("Input folder name for the given datasets")

def convert_file_to_list(file_name):
    with open(data_file+'\\'+file_name) as file:
        file=csv.reader(file,delimiter='\n')
        list_form=[]
        for i in file:
            for j in i:
                j.strip('][')
                list_form.append(j)
        return list_form

train_file=convert_file_to_list('train.csv')
train_file_ns=convert_file_to_list('train_ns.csv')
#lab_train=convert_file_to_list('lab_train.csv')
#print(train_file)
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
lab_train=convert_lab_to_list('lab_train.csv')
#print(lab_train)

def pickle_write(file_name,vector,nb_model):
    with open(file_name,'wb') as pic_file:
        pickle.dump(vector,pic_file)
        pickle.dump(nb_model,pic_file)
        
#--------------------------------------------------------------------------------------------------------------------------------
#Below sections trains the different N gram NB models and stores them in pickle        
#For Unigram
vect_uni=CountVectorizer(ngram_range=(1,1))
vect_uni.fit(train_file)
uni=vect_uni.transform(train_file)
mnb_uni=MultinomialNB().fit(uni,lab_train) # save this output in a pickle file
pickle_write('mnb_uni.pkl',vect_uni,mnb_uni)


#for Bigram
vect_bi=CountVectorizer(ngram_range=(2,2))
vect_bi.fit(train_file)
bi=vect_bi.transform(train_file)
mnb_bi=MultinomialNB().fit(bi,lab_train) # save this output in a pickle file
pickle_write('mnb_bi.pkl',vect_bi,mnb_bi)


#for Unigram and Bigram
vect_uni_bi=CountVectorizer(ngram_range=(1,2))
vect_uni_bi.fit(train_file)
uni_bi=vect_uni_bi.transform(train_file)
mnb_uni_bi=MultinomialNB().fit(uni_bi,lab_train) # save this output in a pickle file
pickle_write('mnb_uni_bi.pkl',vect_uni_bi,mnb_uni_bi)

#Now Generating the same models but for training set without Stopwords. 
#For Unigram
vect_uni_ns=CountVectorizer(ngram_range=(1,1))
vect_uni_ns.fit(train_file_ns)
uni_ns=vect_uni_ns.transform(train_file_ns)
mnb_uni_ns=MultinomialNB().fit(uni_ns,lab_train) # save this output in a pickle file
pickle_write('mnb_uni_ns.pkl',vect_uni_ns,mnb_uni_ns)


#for Bigram
vect_bi_ns=CountVectorizer(ngram_range=(2,2))
vect_bi_ns.fit(train_file_ns)
bi_ns=vect_bi_ns.transform(train_file_ns)
mnb_bi_ns=MultinomialNB().fit(bi_ns,lab_train) # save this output in a pickle file
pickle_write('mnb_bi_ns.pkl',vect_bi_ns,mnb_bi_ns)


#for Unigram and Bigram
vect_uni_bi_ns=CountVectorizer(ngram_range=(1,2))
vect_uni_bi_ns.fit(train_file_ns)
uni_bi_ns=vect_uni_bi_ns.transform(train_file_ns)
mnb_uni_bi_ns=MultinomialNB().fit(uni_bi_ns,lab_train) # save this output in a pickle file
pickle_write('mnb_uni_bi_ns.pkl',vect_uni_bi_ns,mnb_uni_bi_ns)

print("Pickle files have been successfully generated!")


#-----------------------------------------------------------------------------------------------------
#Testing for accuracy.
#loading our test data
test_x=convert_file_to_list('test.csv')
test_x_ns=convert_file_to_list('test_ns.csv')
test_y=convert_lab_to_list('lab_test.csv')

uni_test_trans=vect_uni.transform(test_x)
pred_uni=mnb_uni.predict(uni_test_trans)
print("Accuracy of Unigram: {:.2f}%".format(metrics.accuracy_score(pred_uni, test_y) * 100))
print("Precision of Unigram: {:.2f}%".format(metrics.precision_score(pred_uni, test_y)*100 ))
print("Recall of Unigram: {:.2f}%".format(metrics.recall_score(pred_uni, test_y) * 100))
print("f1 score of Unigram: {:.2f}%".format(metrics.f1_score(pred_uni, test_y) * 100))

bi_test_trans=vect_bi.transform(test_x)
pred_bi=mnb_bi.predict(bi_test_trans)
print("Accuracy of Bigram: {:.2f}%".format(metrics.accuracy_score(pred_bi, test_y) * 100))
print("Precision of Bigram: {:.2f}%".format(metrics.precision_score(pred_bi, test_y) * 100))
print("Recall of Bigram: {:.2f}%".format(metrics.recall_score(pred_bi, test_y) * 100))
print("f1 score of Bigram: {:.2f}%".format(metrics.f1_score(pred_bi, test_y) * 100))

uni_bi_test_trans=vect_uni_bi.transform(test_x)
pred_uni_bi=mnb_uni_bi.predict(uni_bi_test_trans)
print("Accuracy of Unigram and Bigram Combined: {:.2f}%".format(metrics.accuracy_score(pred_uni_bi, test_y) * 100))
print("Precision of Unigram and Bigram Combined: {:.2f}%".format(metrics.precision_score(pred_uni_bi, test_y,pos_label=1) * 100))
print("Recall of Unigram and Bigram Combined: {:.2f}%".format(metrics.recall_score(pred_uni_bi, test_y) * 100))
print("f1 score of Unigram and Bigram Combined: {:.2f}%".format(metrics.f1_score(pred_uni_bi, test_y) * 100))

uni_test_trans_ns=vect_uni_ns.transform(test_x_ns)
pred_uni_ns=mnb_uni_ns.predict(uni_test_trans_ns)
print("Accuracy of Unigram without Stopwords: {:.2f}%".format(metrics.accuracy_score(pred_uni_ns, test_y) * 100))
print("Precision of Unigram without stopwords: {:.2f}%".format(metrics.precision_score(pred_uni_ns, test_y,pos_label=1) * 100))
print("Recall of Unigram without stopwords: {:.2f}%".format(metrics.recall_score(pred_uni_ns, test_y) * 100))
print("f1 score of Unigram without stopwords: {:.2f}%".format(metrics.f1_score(pred_uni_ns, test_y) * 100))

bi_test_trans_ns=vect_bi_ns.transform(test_x_ns)
pred_bi_ns=mnb_bi_ns.predict(bi_test_trans_ns)
print("Accuracy of Bigram without Stopwords: {:.2f}%".format(metrics.accuracy_score(pred_bi_ns, test_y) * 100))
print("Precision of Bigram without stopwords: {:.2f}%".format(metrics.precision_score(pred_bi_ns, test_y,pos_label=1) * 100))
print("Recall of Bigram without stopwords: {:.2f}%".format(metrics.recall_score(pred_bi_ns, test_y) * 100))
print("f1 score of Bigram without stopwords: {:.2f}%".format(metrics.f1_score(pred_bi_ns, test_y) * 100))

uni_bi_test_trans_ns=vect_uni_bi_ns.transform(test_x_ns)
pred_uni_bi_ns=mnb_uni_bi_ns.predict(uni_bi_test_trans_ns)
print("Accuracy of Unigram and Bigram Combined without Stopwords: {:.2f}%".format(metrics.accuracy_score(pred_uni_bi_ns, test_y) * 100))
print("Precision of Unigram and Bigram Combined without stopwords: {:.2f}%".format(metrics.precision_score(pred_uni_bi_ns, test_y,pos_label=1) * 100))
print("Recall of Unigram and Bigram Combined without stopwords: {:.2f}%".format(metrics.recall_score(pred_uni_bi_ns, test_y) * 100))
print("f1_score of Unigram and Bigram Combined without stopwords: {:.2f}%".format(metrics.f1_score(pred_uni_bi_ns, test_y) * 100))
