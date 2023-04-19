#!/usr/bin/env python

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 5000

import glob
import os.path as op
import pickle
import random
import numpy as np

from utils.gradcheck import gradcheck_naive
from utils.utils import softmax

############
# WORD2VEC #
############

def sigmoid(x):
  """
  Compute the sigmoid function for the input here.
  Arguments: x -- A scalar or numpy array.
  Return: s -- sigmoid(x)
  """
  s = 1/(1+ np.exp(-x))
  return s

def naive_softmax_loss_and_gradient(center_word_vec,outside_word_idx,outside_vectors,dataset):
  """ Naive Softmax loss & gradient function for word2vec models

  Implement the naive softmax loss and gradients between a center word's 
  embedding and an outside word's embedding. This will be the building block
  for our word2vec models.

  Arguments:
  center_word_vec -- numpy ndarray, center word's embedding (v_c in the pdf handout)
  outside_word_idx -- integer, the index of the outside word (o of u_o in the pdf handout)
  outside_vectors -- outside vectors (rows of matrix) for all words in vocab (U in the pdf handout)
  dataset -- needed for negative sampling, unused here.

  Return:
  loss -- naive softmax loss
  grad_center_vec -- the gradient with respect to the center word vector (dJ / dv_c in the pdf handout)
  grad_outside_vecs -- the gradient with respect to all the outside word vectors (dJ / dU)
                  
   Note: A softmax() function provided (utils/utils.py) which takes as input a vector/matrix of values and 
        returns the softmax for each value in the vector, relative to the others.

  """
  ### Use the provided softmax function (imported earlier in this file)
  ### This numerically stable implementation helps you avoid issues pertaining to integer overflow.
  
  # Get word probability distribution with respect to v_c
  prob = np.dot(center_word_vec, outside_vectors.T)
  y_hat = softmax(prob)

  # This will also be the change in weights
  delta = y_hat.copy()
  delta[outside_word_idx] -= 1

  # Get context word and calculate its naive softmax loss
  loss = -np.log(y_hat[outside_word_idx])
    
  ### Gradients 
  # Center word gradient & Multiply Jacobians
  grad_center_vec = np.dot(delta, outside_vectors)
    
  # Outside word gradient
  # Cf. outer product of matrix x 'column' vector 
  grad_outside_vecs = np.dot(delta[:, np.newaxis], center_word_vec[np.newaxis, :]) 
  return loss, grad_center_vec, grad_outside_vecs

def get_negative_samples(outside_word_idx, dataset, K):
  """ Samples K indexes which are not the outsideWordIdx """

  neg_sample_word_indices = [None] * K
  for k in range(K):
    newidx = dataset.sample_token_idx()
    while newidx == outside_word_idx:
      newidx = dataset.sample_token_idx()
    neg_sample_word_indices[k] = newidx
  return neg_sample_word_indices

def neg_sampling_loss_and_gradient(center_word_vec,outside_word_idx,outside_vectors,dataset,K=10):
  """ Negative sampling loss function for word2vec models

   Arguments/Return Specifications: same as naive_softmax_loss_and_gradient
   K is the number of negative samples to take.

   """

  neg_sample_word_indices = get_negative_samples(outside_word_idx, dataset, K)
  indices = [outside_word_idx] + neg_sample_word_indices

  grad_center_vec = np.zeros(center_word_vec.shape)
  grad_outside_vecs = np.zeros(outside_vectors.shape)

  labels = np.array([1] + [-1 for k in range(K)])
  vecs = outside_vectors[indices, :]

  t = sigmoid(vecs.dot(center_word_vec) * labels)
  loss = -np.sum(np.log(t))

  delta = labels * (t - 1)
  grad_center_vec = delta.reshape((1, K + 1)).dot(vecs).flatten()
  grad_outside_vecs_temp = delta.reshape((K + 1, 1)).dot(center_word_vec.reshape(
    (1, center_word_vec.shape[0])))
  for k in range(K + 1):
    grad_outside_vecs[indices[k]] += grad_outside_vecs_temp[k, :]

  return loss, grad_center_vec, grad_outside_vecs

def skipgram(current_center_word, window_size, outside_words, word2ind, center_word_vectors, outside_vectors, dataset, word2vec_loss_and_gradient=neg_sampling_loss_and_gradient):
  """ Skip-gram model in word2vec 

  Arguments:
  current_center_word -- a string of the current center word
  window_size -- integer, context window size
  outside_words -- list of no more than 2*window_size strings, the outside words
  word2ind -- a dictionary that maps words to their indices in the word vector list
  center_word_vectors -- center word vectors (as rows) for all words in vocab (V in pdf handout)
  outside_vectors -- outside word vectors (as rows) for all words in vocab (U in pdf handout)
  word2vec_loss_and_gradient -- the loss and gradient function for
                             a prediction vector given the outsideWordIdx
                             word vectors, could be one of the two
                             loss functions you implemented above (do not hardcode any of them).

  Return:
  loss -- the loss function value for the skip-gram model (J in the pdf handout)
  grad_center_vecs -- the gradient with respect to the center word  (dJ / dV in the pdf handout)
  grad_outside_vectors -- the gradient with respect to the outside word vectors (dJ / dU in the pdf handout)
  """

  loss = 0.0
  grad_center_vecs = np.zeros(center_word_vectors.shape)
  grad_outside_vectors = np.zeros(outside_vectors.shape)

  center_word_idx = word2ind[current_center_word]
  center_word_vec = center_word_vectors[center_word_idx]
    
  # Iterate over outside words
  for word in outside_words:
      outside_word_idx = word2ind[word]
      current_loss, grad_center_vec, grad_outside_vecs = \
          word2vec_loss_and_gradient(center_word_vec, outside_word_idx,
                                     outside_vectors, dataset)
      loss += current_loss
      grad_center_vecs[center_word_idx] += grad_center_vec
      grad_outside_vectors += grad_outside_vecs

  return loss, grad_center_vecs, grad_outside_vectors

def word2vec_sgd_wrapper(word2vec_model, word2ind, word_vectors, dataset, window_size, word2vec_loss_and_gradient=neg_sampling_loss_and_gradient):
  batchsize = 50
  loss = 0.0
  grad = np.zeros(word_vectors.shape)
  N = word_vectors.shape[0]
  center_word_vectors = word_vectors[:int(N / 2), :]
  outside_vectors = word_vectors[int(N / 2):, :]
  for i in range(batchsize):
    window_size_1 = random.randint(1, window_size)
    center_word, context = dataset.get_random_context(window_size_1)

    c, gin, gout = word2vec_model(center_word, window_size_1, context, word2ind, center_word_vectors,outside_vectors, dataset, word2vec_loss_and_gradient)
    loss += c / batchsize
    grad[:int(N / 2), :] += gin / batchsize
    grad[int(N / 2):, :] += gout / batchsize

  return loss, grad

#######
# SGD #
#######

def load_saved_params():
    """
    A helper function that loads previously saved parameters and resets iteration start.
    """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter

    if st > 0:
        params_file = "saved_params_%d.npy" % st
        state_file = "saved_state_%d.pickle" % st
        params = np.load(params_file)
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None

def save_params(iter, params):
    params_file = "saved_params_%d.npy" % iter
    np.save(params_file, params)
    with open("saved_state_%d.pickle" % iter, "wb") as f:
        pickle.dump(random.getstate(), f)

def sgd(f, x0, step, iterations, postprocessing=None, use_saved=False,PRINT_EVERY=10):
  """ Stochastic Gradient Descent 

  Arguments:
  f -- the function to optimize, it should take a single
       argument and yield two outputs, a loss and the gradient
       with respect to the arguments
  x0 -- the initial point to start SGD from
  step -- the step size for SGD
  iterations -- total iterations to run SGD for
  postprocessing -- postprocessing function for the parameters
                    if necessary. In the case of word2vec we will need to
                    normalize the word vectors to have unit length.
  PRINT_EVERY -- specifies how many iterations to output loss

  Return: x -- the parameter value after SGD finishes
  """

  # Anneal learning rate every several iterations
  ANNEAL_EVERY = 20000

  if use_saved:
    start_iter, oldx, state = load_saved_params()
    if start_iter > 0:
      x0 = oldx
      step *= 0.5 ** (start_iter / ANNEAL_EVERY)

    if state:
      random.setstate(state)
  else:
    start_iter = 0

  x = x0

  if not postprocessing:
    postprocessing = lambda x: x

  exploss = None

  for iter in range(start_iter + 1, iterations + 1):
    loss = None
    loss, grad = f(x)
    x -= step * grad

    x = postprocessing(x)
    if iter % PRINT_EVERY == 0:
      if not exploss:
        exploss = loss
      else:
        exploss = .95 * exploss + .05 * loss
      print("iter %d: %f" % (iter, exploss))

    if iter % SAVE_PARAMS_EVERY == 0 and use_saved:
      save_params(iter, x)

    if iter % ANNEAL_EVERY == 0:
      step *= 0.5
  return x