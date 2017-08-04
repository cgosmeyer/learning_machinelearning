# CHAPTER 7: WORKING WITH TEXT DATA

see jupyter notebook `ch7.ipynb`.

Four kinds of string data:

1. Categorical data (colors, Iris species, etc)

2. Free strings that can be semantically mapped to categories

3. Structured string data (addresses, names of people, dates)

4. Text data (tweets, hotel reviews, work of Shakespeare)

## REPRESENTING TEXT DATA AS A BAG OF WORDS

In this representation we discard most of the structure of the input text, like chapters, paragraphs, sentences, and formatting, and *only count how often each word appears in each text.*  Note that the order of words is irrelevant.

For small datasets, it might be worth discarding *stop words*, that is, words like "the", "however", "beforehand".

Instead of dropping features that are deemed unimportant, coould rescale features by how informative we expect them to be. One common way is using *term frequency-inverse document frequency* (tf-idf). If a word appears often in a particular document, but not in very many documents, it is likely to be very descriptive of the content of that document. 

Bag of words can be improved by considering pairs (bigrams) or triplets (trigrams) of tokens that appear next to eacher, such as "not good".  But using more than 2 grams can lead to overfitting, since there will be many very specific features.

To avoid overfitting from "replace", "recplaced", "replacement", etc., represent each word with its *word stem*. Taking role of the word in a sentence into account is called *lemmatization*. A third option is to try to correct spelling errors.

## TOPIC MODELING AND DOCUMENT CLUSTERING

Topic modeling is assigning each document to one or multiple topics, usually without supervision. In news articles, could have "sports", "politics", "finance". One particular decomposition method that is used is *Latent Dirichlet Allocation* (LDA).
