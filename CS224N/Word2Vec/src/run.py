#!/usr/bin/env python

import os
import random
import matplotlib
import numpy as np

from utils.treebank import StanfordSentiment
from utils.utils import dump

matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

from model import *
import sys

assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
n_words = len(tokens)

# Train 10-dimensional vectors for this project
dim_vectors = 10
C = 5 # Context size

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)

start_time = time.time()

word_vectors = np.concatenate(
    ((np.random.rand(n_words, dim_vectors) - 0.5) /
     dim_vectors, np.zeros((n_words, dim_vectors))),
    axis=0)

word_vectors = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
                                     neg_sampling_loss_and_gradient),
    word_vectors, 0.3, 40000, None, False, PRINT_EVERY=10)

# Note that normalization is not called here. This is not a bug,
# normalizing during training loses the notion of length.

print("sanity check: cost at convergence should be around or below 10")
print("training took %d seconds" % (time.time() - start_time))

# Concatenate the input and output word vectors
word_vectors = np.concatenate(
    (word_vectors[:n_words, :], word_vectors[n_words:, :]),
    axis=0)

visualize_words = [
    "great", "cool", "brilliant", "wonderful", "well", "amazing",
    "worth", "sweet", "enjoyable", "boring", "bad", "dumb",
    "annoying", "female", "male", "queen", "king", "man", "woman", "rain", "snow",
    "hail", "coffee", "tea"]

# dimensionality reduction
visualize_idx = [tokens[word] for word in visualize_words]
visualize_vecs = word_vectors[visualize_idx, :]

# save word vectors for evaluation
sample_vectors = {word: list(vec) for word, vec in zip(visualize_words, visualize_vecs)}
sample_vectors_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sample_vectors_(soln).json")
dump(sample_vectors, sample_vectors_path)

temp = (visualize_vecs - np.mean(visualize_vecs, axis=0))
covariance = 1.0 / len(visualize_idx) * temp.T.dot(temp)
U, S, V = np.linalg.svd(covariance)
coord = temp.dot(U[:, 0:2])

for i in range(len(visualize_words)):
    plt.text(coord[i, 0], coord[i, 1], visualize_words[i],
             bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))

plt.savefig('word_vectors_(soln).png')
