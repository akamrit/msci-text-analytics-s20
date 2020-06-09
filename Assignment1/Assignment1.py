# -*- coding: utf-8 -*-

import re
import csv
import random

#reading dataset file
rd=open('C:/Users/Amrit Kaur/Desktop/assn1/pos.txt')
file1=rd.read()
#print(file1)

#tokenizing the corpus
token=file1.splitlines()
#print(token)

for i in range(len(token)):
    token[i]=token[i].splitlines()
    

#removing special characters.
new_list1=re.sub('[^A-Za-z0-9. \n]', '',file1)
token1=new_list1.splitlines()
#This is the list without removing stop words(token1).
#print(token1)
for i in range(len(token1)):
    token1[i]=token1[i].split()
#creating a list without stopwords
scnd_list=[[]]

stopword=["a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"]

for i in range (len(token1)):
    l=[]
    for j in range (len(token1[i])):
        if token1[i][j] not in stopword:
            l.append(token1[i][j])
    scnd_list.append(l)
            
for i in range(len(scnd_list)):
    print(i,scnd_list[i])

    
# writing the list with stop words to csv file  
with open("C:/Users/Amrit Kaur/Desktop/assn1/out1.csv", mode='w', newline='') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
    # writing the data rows  
    for i in token1:
        csvwriter.writerow([i]) 

# opening csv file with stopwords      
r=open('C:/Users/Amrit Kaur/Desktop/assn1/out1.csv')
a=r.read()
#print(a)

# writing the list without stop words to csv file  
with open("C:/Users/Amrit Kaur/Desktop/assn1/out2.csv", mode='w', newline='') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)   
        
# writing the data rows  
    for i in scnd_list:
        csvwriter.writerow([i]) 
    
# writing the list without stop words to csv file  
c=open('C:/Users/Amrit Kaur/Desktop/assn1/out2.csv')
b=c.read()
#print(b)


#print(token1)
random.shuffle(token1)
#print(token1)

#dividing the dataset into 80% training, 10% validatation, 10% testing

training_with_sw = token1[:int(len(token1)*0.8)] #[1, 2, 3, 4, 5, 6, 7, 8]
validation_with_sw = token1[int(len(token1)*0.8):int(len(token1)*0.9)] #[10]
testing_with_sw = token1[-int(len(token1)*0.1):] #[10]


#output of training dataset with stopwords
#print(training_with_sw)

with open("C:/Users/Amrit Kaur/Desktop/assn1/out3.csv", mode='w', newline='') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
        
        
    # writing the data rows  
    for i in training_with_sw:
        csvwriter.writerow([i]) 
    
#output of validation dataset with stopwords    
#print(validation_with_sw)    
with open("C:/Users/Amrit Kaur/Desktop/assn1/out4.csv", mode='w', newline='') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
        
    # writing the data rows  
    for i in validation_with_sw:
        csvwriter.writerow([i]) 

    
#output of testing dataset with stopwords    
#print(testing_with_sw)
with open("C:/Users/Amrit Kaur/Desktop/assn1/out5.csv", mode='w', newline='') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
        
    # writing the data rows
    for i in testing_with_sw:
        csvwriter.writerow([i]) 
 
    
#dividing the dataset without stopwords into 80% training,10%testing,10% validation  
training_without_sw = scnd_list[:int(len(scnd_list)*0.8)] #[1, 2, 3, 4, 5, 6, 7, 8]
validation_without_sw = scnd_list[int(len(scnd_list)*0.8):int(len(scnd_list)*0.9)] #[10]
testing_without_sw = scnd_list[-int(len(scnd_list)*0.1):] #[10]

#print(training_without_sw)



with open("C:/Users/Amrit Kaur/Desktop/assn1/out6.csv", mode='w', newline='') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)      
        
    # writing the data rows 
    for i in training_without_sw:
        csvwriter.writerow([i])

    
    
#print(validation_with_sw)    
with open("C:/Users/Amrit Kaur/Desktop/assn1/out7.csv", mode='w', newline='') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
        
        
    # writing the data rows 
    for i in validation_without_sw:
        csvwriter.writerow([i])
    
#print(testing_with_sw)
with open("C:/Users/Amrit Kaur/Desktop/assn1/out8.csv", mode='w', newline='') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
        
        
    # writing the data rows 
    for i in testing_without_sw:
        csvwriter.writerow([i])

    

    

    
    



