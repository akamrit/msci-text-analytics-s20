# MSCI 641 Assignment

* The output words that are produced as similar to the word ‘good’ are most likely positive with a 10% exception. There are certain words like ‘superb’,’ fabulous’,’ excellent’ that resembles positive words in corpus. Similarly, some of the words that are similar to bad are- ‘horrible’,’poor’,’pathetic’. The word2vec model searches for the words that are related to the word ‘good’ and ‘bad’ respectively and since bad is a negative word, we expect output vector to produce most likely negative words and vice-versa.

* This may be the case as the output vector find those words that are closely lying around the input vector. The output words get closer vector positions because of the context they appear in corpus and because they lie in the same cluster and hence their cosine angles are the same. The exception of some words might exist as they have similar vectors as input vector because they might have used somewhere in the same context to describe a thing.
