from nltk.corpus import wordnet

from sklearn import linear_model
import os
import numpy as np
import pandas as pd
import enchant
dictionary = enchant.Dict('en_US')
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
nltk.download('brown')
nltk.download('universal_tagset')
import re

reviewsDF = pd.read_csv("parsed_workable-1000.csv")
reviewsDF.drop(columns=['business_id'], inplace=True)
# print(list(reviewsDF.columns))
# reviewsDF = reviewsDF.head(200)

# stemmer = SnowballStemmer("english")
# reviewsDF['stemmed'] = reviewsDF['text'].map(lambda x: ' '.join([stemmer.stem(y) for y in word_tokenize(x)]))
# print(reviewsDF.stemmed.head())

cvec = CountVectorizer(stop_words='english', min_df=1, max_df=.5, ngram_range=(1,1))

from itertools import islice
cvec.fit(reviewsDF.text)
# print(list(islice(cvec.vocabulary_.items(), 20)))
# print(len(cvec.vocabulary_))

cvec_counts = cvec.transform(reviewsDF.text)
occ = np.asarray(cvec_counts.sum(axis=0)).ravel().tolist()
counts_df = pd.DataFrame({'term': cvec.get_feature_names(), 'occurrences': occ})
counts_df.sort_values(by='occurrences', ascending=False).head(20)

transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(cvec_counts)

weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
# print(weights_df.sort_values(by='weight', ascending=False).head(20))

lexiconDF = pd.read_csv(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vader_lexicon.txt')), sep='\t', header=None, names=('token', 'polarity', 'sentiment', 'list'))
lexiconDF.drop(columns=['sentiment', 'list'], inplace=True)
print(len(lexiconDF))

# print(list(lexiconDF.columns))
# keys = list(lexiconDF.columns.token)
# i1 = reviewsDF.set_index(keys).index
# i2 = lexiconDF.set_index(keys).index
# print("LEXICON")
# print(lexiconDF.head(10))
# common = weights_df[(weights_df.term.isin(lexiconDF.token))].sort_values(by='weight', ascending=False)
# print(common.head(10))
common = lexiconDF.merge(weights_df, left_on="token", right_on="term").dropna()
common.drop(columns=['term'], inplace=True)
# print(common.columns.values)
wordtags = nltk.ConditionalFreqDist((w.lower(), t)
        for w, t in nltk.corpus.brown.tagged_words(tagset="universal"))

regSymbols = r'[^a-zA-Z ]'
# common["synonym"] = common.term.apply(lambda word: list(set([
#     re.sub(regSymbols, '', str(item.lower())) for sublist in [w.lemma_names() for w in wordnet.synsets(word)]
#     for item in sublist if '_' not in item
# ])))
common["synonym"] = common.token.apply(lambda word: list(set([
    re.sub(regSymbols, '', str(item.lower())) for sublist in [w.lemma_names() for w in wordnet.synsets(word)]
    for item in sublist if ('ADJ' in list(wordtags[item])) and (item != word)
])))
# pos_common = common[common['polarity'] > 0].sort_values(by='weight', ascending=False)
# print(pos_common.head(10))
# print()
# neg_common = common[common['polarity'] < 0].sort_values(by='weight', ascending=False)
# print(neg_common.head(10))
# print()
pos_syn = {}
remove = set()
def unra(row):
    for word in row['synonym']:
        if word in pos_syn:
            remove.add(word)
        else:
            pos_syn[word] = row['polarity']
common.apply(lambda row: unra(row), axis=1)
for word in remove:
    del pos_syn[word]
syn_df = pd.DataFrame.from_dict(pos_syn, orient='index', columns=['polarity']).reset_index()
syn_df.columns = ['token', 'polarity']
# print(syn_df)
# dif = syn_df[~syn_df.index.isin(lexiconDF.token)]
print(len(set(lexiconDF['token']).union(set(syn_df['token']))))

# words found in both vader and new synonym lex
# mergedStuff = lexiconDF.merge(syn_df, left_on='token', right_on ='index', how='inner')
# mergedStuff.drop(columns=['index'], inplace=True)
# mergedStuff.columns = ['token','vader_pol','syn_pol']
# print(mergedStuff)
dfNew = lexiconDF.append(syn_df, ignore_index=True).drop_duplicates(subset=['token'], keep='first')
print(dfNew)
# mergedStuff = syn_df.merge(lexiconDF, left_on='token', right_on ='token', how='outer').drop_duplicates()
# print(mergedStuff)
dfNew.to_csv(r'amanda_oov-1000.csv', index=False, header=True)

# neg_common = common[common['polarity'] < 0].sort_values(by='weight', ascending=False)
# print(neg_common.head(10))
# for index, row in common.head(10).iterrows():
#     for word in row['synonym']:
#         print(word)
