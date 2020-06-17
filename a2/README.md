# MSCI 641 Assignment

The Classification accuracies are given below-
 * Stopword Removed	Text Features	Accuracy(test set)
 * Yes	unigrams	80.67%
 * Yes	bigrams	82.39%
 * Yes	Unigrams + bigrams	83.12%
 * No	unigrams	80.48%
 * No	bigrams	78.21%
 * No	Unigrams + bigrams	82.15%

* Stopword removal – As it can be seen from the results that the corpus in which stop words were removed performed slightly better than the counter part. This is due to the fact that stopwords add to the total word count which are redundant in nature and decreases probabiliteis of the words which are actually required. However this varies from case to case basis as it is beneficial in large datasets but will decrease the efficiency in small datasets.
* Unigram/Bigram/Unigram+Bigram-  From our experiment, we see that as the N-grams increases , the performance is becoming better. We can see the performance pattern is more likely Unigram+Bigram >Bigram >Unigram. However, it depends on corpus too. We see that there is slight fluctuation in bigram. The more the words combined together, the better is the result for prediction.
