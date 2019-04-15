#!/usr/bin/env python

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils.glove import loadWordVectors
from q2g_knn import *
import numpy as np

visualizeWords = [
    "the", "a", "an", ",", ".", "?", "!", "``", "''", "--",
    "good", "great", "cool", "brilliant", "wonderful", "well", "amazing",
    "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb",
    "annoying"]

key_words = ["the", "unique", "superb", "comedy", "surprisingly"]

tokens = {}
ind = 0
for word in visualizeWords + key_words:
    if word not in tokens:
        tokens[word] = ind
        ind += 1

wordVectors = loadWordVectors(tokens)

visualizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2])

for i in xrange(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i],
        bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('glove_word_vectors.png')

inv_tokens = {v: k for k, v in tokens.iteritems()}
for key_word in key_words:
    wordVector = wordVectors[tokens[key_word]]
    idx = knn(wordVector, wordVectors, 11)
    print "Words related to \"" + key_word + "\": ",  [inv_tokens[i] for i in idx]

