import csv
import os
import time
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow
import cloudpickle
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from keras.layers import Dense,Dropout, LSTM
import keras
import numpy
from gensim.models import Word2Vec
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import naive_bayes, linear_model, svm
from sklearn.neighbors import KNeighborsClassifier
import joblib
import gc
import platform
import os
import sys

HIDDEN_LAYER_SIZE = 100

DROPOUT = 0.5
L2_LAMBDA = 0.001           
LEARNING_RATE = 0.00146 

TRAIN_BATCH_SIZE = 500
EPOCHS = 40

FIELDNAMES = ['Headline', 'Body ID', 'Stance']
LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
RELATED = LABELS[0:3]

def score_submission(gold_labels, test_labels):
    score = 0.0
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        g_stance, t_stance = g, t
        if g_stance == t_stance:
            score += 0.25
            if g_stance != 'unrelated':
                score += 0.50
        if g_stance in RELATED and t_stance in RELATED:
            score += 0.25

        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

    return score, cm

def report_score(actual,predicted):
    score,cm = score_submission(actual,predicted)
    best_score, _ = score_submission(actual,actual)

    print_confusion_matrix(cm)
    print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return score*100/best_score

def print_confusion_matrix(cm):
    lines = []
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                   *row))
        lines.append("-"*line_len)
    print('\n'.join(lines))

def simple_neural_network(X_train, y_train, X_test, actual_stances):
    model = keras.Sequential()
    
    # model.add(Dense(HIDDEN_LAYER_SIZE, activation = 'relu', input_dim = len(X_train[0])))
    # model.add(Dropout(DROPOUT))
    
    model.add(Dense(HIDDEN_LAYER_SIZE, activation = 'relu', kernel_regularizer=regularizers.l2(L2_LAMBDA)))
    model.add(Dropout(DROPOUT))
    model.add(Dense(4, activation = 'softmax'))

    adam = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer= adam, loss='categorical_crossentropy', metrics= ['accuracy'])
    y_train_categorical = keras.utils.to_categorical(numpy.asarray(y_train))
    model.fit(X_train, y_train_categorical, epochs = EPOCHS, batch_size= TRAIN_BATCH_SIZE)
    predictions = model.predict_classes(X_test)

    with open(PATH_TRAIN_PICKLE, 'rb') as file:
        label_encoder = joblib.load(file)['label_encoder']

    predicted_stances = label_encoder.inverse_transform(predictions)

    print('Simple neural network:')
    score = report_score(actual_stances, predicted_stances)

    return score, predicted_stances

def write_answer_csv(predictions):

    unlabelled_data  = open(TEST_FILE_UNLABELLED, 'r', encoding='utf-8')
    unlabelled_reader = csv.DictReader(unlabelled_data)

    write_file = open(ANSWER_FILE, 'w', newline= '', encoding= 'utf-8')

    writer = csv.DictWriter(write_file, fieldnames=['Headline','Body ID','Stance'])
    writer.writeheader()
    for unlabelled, prediction in zip(unlabelled_reader, predictions.tolist()):
        writer.writerow({'Body ID': unlabelled['Body ID'],'Headline': unlabelled['Headline'],'Stance': prediction})

def K_Nereast_Neighbour(X_train,y_train,X_test, actual_stances):
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    with open(PATH_TRAIN_PICKLE, 'rb') as file:
        label_encoder = cloudpickle.load(file)['label_encoder']

    predicted_stances = label_encoder.inverse_transform(predictions)

    print('KNN:')
    score = report_score(actual_stances, predicted_stances)

    return score, predicted_stances



def Naive_Bayes_Classifier(X_train,y_train,X_test, actual_stances):
    classifier = naive_bayes.MultinomialNB()
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    with open(PATH_TRAIN_PICKLE, 'rb') as file:
        label_encoder = cloudpickle.load(file)['label_encoder']

    predicted_stances = label_encoder.inverse_transform(predictions)

    print('Naive Bayes:')
    score = report_score(actual_stances, predicted_stances)

    return score, predicted_stances

def Decision_Tree_Classifier(X_train,y_train,X_test, actual_stances):
    classifier = DecisionTreeClassifier(random_state=0)
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    with open(PATH_TRAIN_PICKLE, 'rb') as file:
        label_encoder = cloudpickle.load(file)['label_encoder']

    predicted_stances = label_encoder.inverse_transform(predictions)

    print('Decision Tree Classifier:')
    score = report_score(actual_stances, predicted_stances)

    return score, predicted_stances

def Random_Forest_Classifier(X_train,y_train,X_test, actual_stances):
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    with open(PATH_TRAIN_PICKLE, 'rb') as file:
        label_encoder = joblib.load(file)['label_encoder']

    predicted_stances = label_encoder.inverse_transform(predictions)

    print('Random Forest Classifier:')
    score = report_score(actual_stances, predicted_stances)

    return score, predicted_stances

def Gradient_Boosting_Classifier(X_train,y_train,X_test, actual_stances):
    classifier = GradientBoostingClassifier(n_estimators=20, random_state=14128, verbose=True)
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    with open(PATH_TRAIN_PICKLE, 'rb') as file:
        label_encoder = cloudpickle.load(file)['label_encoder']

    predicted_stances = label_encoder.inverse_transform(predictions)

    print('Gradient Boosting Classifier:')
    score = report_score(actual_stances, predicted_stances)

    return score, predicted_stances

# def XGBoostClassifier(X_train,y_train,X_test, actual_stances):
#     xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
#     xg_reg.fit(X_train,y_train)

#     predictions = xg_reg.predict(X_test)

#     with open(PATH_TRAIN_PICKLE, 'rb') as file:
#         label_encoder = cloudpickle.load(file)['label_encoder']

#     predicted_stances = label_encoder.inverse_transform(predictions)

#     print('Gradient Boosting Classifier:')
#     score = report_score(actual_stances, predicted_stances)

#     return score, predicted_stances

def Linear_Classifier(X_train,y_train,X_test, actual_stances):
    classifier = linear_model.LogisticRegression()
    classifier.fit(X_train, y_train)

    tf.optimizers

    predictions = classifier.predict(X_test)

    with open(PATH_TRAIN_PICKLE, 'rb') as file:
        label_encoder = cloudpickle.load(file)['label_encoder']

    predicted_stances = label_encoder.inverse_transform(predictions)

    print('Linear Classifier:')
    score = report_score(actual_stances, predicted_stances)

    return score, predicted_stances

def Support_Vector_Machine(X_train,y_train,X_test, actual_stances):
    classifier = svm.LinearSVC()
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    with open(PATH_TRAIN_PICKLE, 'rb') as file:
        label_encoder = cloudpickle.load(file)['label_encoder']

    predicted_stances = label_encoder.inverse_transform(predictions)

    print('Support Vector Machine:')
    score = report_score(actual_stances, predicted_stances)

    return score, predicted_stances

def main(data_path):

    path = data_path
    global PATH_TRAIN_PICKLE
    PATH_TRAIN_PICKLE = os.path.join(path, 'dataset_train_encoded.pyk')
    global PATH_TEST_PICKLE
    PATH_TEST_PICKLE  = os.path.join(path, 'dataset_test_encoded.pyk')
    global COMPETITION_TEST_STANCES
    COMPETITION_TEST_STANCES = os.path.join(path, 'competition_test_stances.csv')
    global TEST_FILE_UNLABELLED
    TEST_FILE_UNLABELLED = os.path.join(path, 'competition_test_stances_unlabeled.csv')
    global ANSWER_FILE
    ANSWER_FILE = os.path.join(path, 'answer.csv')

    logging.info('Loading train set')
    with open(PATH_TRAIN_PICKLE, 'rb') as file:
        stash = joblib.load(file)
        X_train = np.asarray(stash['X'])
        y_train = np.asarray(stash['y'])

    logging.info('Loading test set')
    with open(PATH_TEST_PICKLE, 'rb') as file:
        X_test = joblib.load(file)['X']
        X_test = np.asarray(X_test)

    test_stance = pd.read_csv(COMPETITION_TEST_STANCES)
    actual_stances = test_stance['Stance']

    # score, predictions = simple_neural_network(X_train, y_train, X_test, actual_stances)

    # score,predictions = Decision_Tree_Classifier(X_train, y_train, X_test, actual_stances)
    score,predictions = Random_Forest_Classifier(X_train, y_train, X_test, actual_stances)
    # score,predictions = K_Nereast_Neighbour(X_train, y_train, X_test, actual_stances)
    # score,predictions = Linear_Classifier(X_train, y_train, X_test, actual_stances)
    # score,predictions = Support_Vector_Machine(X_train, y_train, X_test, actual_stances)
    write_answer_csv(predictions)

    print('End of Code')

if __name__ == "__main__":
    main(sys.argv[1])