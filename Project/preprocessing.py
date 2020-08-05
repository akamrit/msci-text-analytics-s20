import csv

import cloudpickle
import pickle
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
import re
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from nltk.corpus import stopwords
import os
import sys

VOCABULARY_SIZE   = 5000

def get_or_put(list_h, value):
    try:
        return list_h.index(value)
    except ValueError:
        list_h.append(value)
        return len(list_h) - 1

def clean_data(line):
    dele_words = '(,|\n|\.|\d|\'|-)'
    wn = nltk.WordNetLemmatizer()
    stop_words = stopwords.words('english')
    line = re.sub(dele_words, ' ', line)
    line = re.sub('\s+', ' ', line)
    word_tokens = nltk.word_tokenize(line)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    lm_filtered_sentence = [wn.lemmatize(word) for word in filtered_sentence]
    line = ' '.join(lm_filtered_sentence)
    return line

def read_stance(filepath, has_labels=True):
    samples = list()
    headlines = list()

    with open(filepath, 'r', encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for line in reader:
            headline = line['Headline']
            headline = clean_data(headline.lower())
            hid = get_or_put(headlines, headline)

            node = {
                'headline': hid,
                'body_id': int(line['Body ID']),
            }

            if has_labels:
                node['label'] = line['Stance']

            samples.append(node)

    return samples, headlines


def read_bodies(filepath):
    ordered_bodies = list()
    bodies_index = dict()

    with open(filepath, 'r', encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for line in reader:
            body_id = int(line['Body ID'])
            body = line['articleBody']
            body = clean_data(body.lower())

            if body_id not in bodies_index:
                ordered_bodies.append(body)
                bodies_index[body_id] = len(ordered_bodies) - 1

    return ordered_bodies, bodies_index

def build_vector(tf_head, tf_body, similarity1, similarity2):
    return \
    np.concatenate((tf_head.toarray(), tf_body.toarray(), similarity1, similarity2), axis=1)[
        0]

def save_model(filepath, X, y=None, encoder=None):
    data = dict(X=X)
    if y is not None:
        data['y'] = y
    if encoder:
        data['label_encoder'] = encoder

    with open(filepath, 'wb') as file:
        # cloudpickle.dump(data, file)
        joblib.dump(data, file) 


def main(data_path):
    path = data_path

    path_train_stances = os.path.join(path, 'train_stances.csv')
    path_train_bodies  = os.path.join(path,'train_bodies.csv')
    path_test_stances  = os.path.join(path,'test_stances_unlabeled.csv')
    path_test_bodies   = os.path.join(path,'test_bodies.csv')

    path_store_train   = os.path.join(path,'dataset_train_encoded.pyk')
    path_store_test    = os.path.join(path,'dataset_test_encoded.pyk')

    print('Reading data')
    train_samples, train_headlines = read_stance(path_train_stances)
    train_bodies, train_bodies_map = read_bodies(path_train_bodies)

    test_samples, test_headlines = read_stance(path_test_stances, False)
    test_bodies, test_bodies_map = read_bodies(path_test_bodies)


    print('Fitting and transforming TF vectorizer.')
    all_train_texts = train_headlines + train_bodies

    tf_vectorizer = TfidfVectorizer(max_features=VOCABULARY_SIZE, use_idf=False)
    all_tfs = tf_vectorizer.fit_transform(all_train_texts)

    print('Fitting and transforming TF-IDF vectorizer.')
    all_texts = all_train_texts + test_headlines + test_bodies
    tfidf_transformer = TfidfVectorizer(max_features=VOCABULARY_SIZE)
    _ = tfidf_transformer.fit(all_texts)

    first_headline = len(train_headlines)

    print('Fitting label encoder')
    all_labels = list(map(lambda sample: sample['label'], train_samples))
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)

    train_feature_samples = list()
    for sample in train_samples:
        headline_id = sample['headline']
        body_id = sample['body_id']

        body_index = train_bodies_map[body_id]

        tf_headline = all_tfs[headline_id]
        tf_body = all_tfs[first_headline + body_index]

        tf_idf_head = tfidf_transformer.transform([train_headlines[headline_id]])
        tf_idf_body = tfidf_transformer.transform([train_bodies[body_index]])

        s = cosine_similarity(tf_idf_head, tf_idf_body, dense_output=True)
        e = euclidean_distances(tf_idf_head, tf_idf_body)

        train_feature_samples.append(build_vector(tf_headline, tf_body, s, e))

    headline_tf_2 = dict()
    body_tf_2 = dict()

    test_feature_samples = list()
    for sample in test_samples:
        headline_id = sample['headline']
        body_id = sample['body_id']

        body_index = test_bodies_map[body_id]

        if headline_id not in headline_tf_2:
            text = [test_headlines[headline_id]]

            tf_headline = tf_vectorizer.transform(text)
            tfidf_headline = tfidf_transformer.transform(text)

            headline_tf_2[headline_id] = (tf_headline, tfidf_headline)

        tf_headline, tfidf_headline = headline_tf_2.get(headline_id)

        if body_index not in body_tf_2:
            text = [test_bodies[body_index]]

            tf_body = tf_vectorizer.transform(text)
            tfidf_body = tfidf_transformer.transform(text)

            body_tf_2[body_index] = (tf_body, tfidf_body)

        tf_body, tfidf_body = body_tf_2.get(body_index)

        s = cosine_similarity(tfidf_headline, tfidf_body)
        e = euclidean_distances(tfidf_headline, tfidf_body)
        
        test_feature_samples.append(build_vector(tf_headline, tf_body, s, e))

    print('Saving trained model')
    save_model(path_store_train, train_feature_samples, encoded_labels, label_encoder)

    print('Saving test data.')
    save_model(path_store_test, test_feature_samples)

if __name__ == "__main__":
    main(sys.argv[1])