#!/usr/bin/env python3

import unittest, random, sys, copy, argparse, inspect
from testerUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np
from utils.treebank import *
from utils.utils import load
from utils.gradcheck import gradcheck_naive
from utils.utils import softmax
from utils.utils import normalize_rows
from scipy import spatial
import os
import traceback

import model

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################
def dummy():
    random.seed(31415)
    np.random.seed(9265)

    dataset = type('dummy', (), {})()

    def dummy_sample_token_idx():
        return random.randint(0, 4)

    def get_random_context(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
               [tokens[random.randint(0, 4)] for i in range(2 * C)]

    dataset.sample_token_idx = dummy_sample_token_idx
    dataset.get_random_context = get_random_context

    dummy_vectors = normalize_rows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    return dataset, dummy_vectors, dummy_tokens

inputs = {
    'test_word2vec': {
        'current_center_word': "c",
        'window_size': 3,
        'outside_words': ["a", "b", "e", "d", "b", "c"]
    },
    'test_naivesoftmax': {
        'center_word_vec': np.array([-0.27323645, 0.12538062, 0.95374082]).astype(float),
        'outside_word_idx': 3,
        'outside_vectors': np.array([[-0.6831809, -0.04200519, 0.72904007],
                                    [0.18289107, 0.76098587, -0.62245591],
                                    [-0.61517874, 0.5147624, -0.59713884],
                                    [-0.33867074, -0.80966534, -0.47931635],
                                    [-0.52629529, -0.78190408, 0.33412466]]).astype(float)

    },
    'test_sigmoid': {
        'x': np.array([-0.46612273, -0.87671855, 0.54822123, -0.36443576, -0.87671855, 0.33688521
                          , -0.87671855, 0.33688521, -0.36443576, -0.36443576, 0.54822123]).astype(float)
    }
}

outputs = {
    'test_word2vec': {
        'loss': 11.16610900153398,
        'dj_dv': np.array(
            [[0., 0., 0.],
             [0., 0., 0.],
             [-1.26947339, -1.36873189, 2.45158957],
             [0., 0., 0.],
             [0., 0., 0.]]).astype(float),
        'dj_du': np.array(
            [[-0.41045956, 0.18834851, 1.43272264],
             [0.38202831, -0.17530219, -1.33348241],
             [0.07009355, -0.03216399, -0.24466386],
             [0.09472154, -0.04346509, -0.33062865],
             [-0.13638384, 0.06258276, 0.47605228]]).astype(float)

    },
    'test_naivesoftmax': {
        'loss': 2.217424877675181,
        'dj_dvc': np.array([-0.17249875, 0.64873661, 0.67821423]).astype(float),
        'dj_du': np.array([[-0.11394933, 0.05228819, 0.39774391],
                           [-0.02740743, 0.01257651, 0.09566654],
                           [-0.03385715, 0.01553611, 0.11817949],
                           [0.24348396, -0.11172803, -0.84988879],
                           [-0.06827005, 0.03132723, 0.23829885]]).astype(float)
    },
    'test_sigmoid': {
        's': np.array(
            [0.38553435, 0.29385824, 0.63372281, 0.40988622, 0.29385824, 0.5834337, 0.29385824, 0.5834337, 0.40988622,
             0.40988622, 0.63372281]).astype(float),
    }
}

sample_vectors_expected = {
    "female": [
        0.6029723815239835,
        0.16789318536724746,
        0.22520087305967568,
        -0.2887330648792561,
        -0.914615719505456,
        -0.2206997036383445,
        0.2238454978107194,
        -0.27169214724889107,
        0.6634932978039564,
        0.2320323110106518
    ],
    "cool": [
        0.5641256072125872,
        0.13722982658305444,
        0.2082364803517175,
        -0.2929695723456364,
        -0.8704480862547578,
        -0.18822962799771015,
        0.24239616047158674,
        -0.29410091959922546,
        0.6979644655991716,
        0.2147529764765611
    ]
}

def test_naive_softmax_loss_and_gradient():
    print("\t\t\tnaive_softmax_loss_and_gradient\t\t\t")

    dataset, dummy_vectors, dummy_tokens = dummy()

    print("\nYour Result:")
    loss, dj_dvc, dj_du = model.naive_softmax_loss_and_gradient(
        inputs['test_naivesoftmax']['center_word_vec'],
        inputs['test_naivesoftmax']['outside_word_idx'],
        inputs['test_naivesoftmax']['outside_vectors'],
        dataset
    )

    print(
        "Loss: {}\nGradient wrt Center Vector (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(loss,
                                                                                                                  dj_dvc,
                                                                                                                  dj_du))

    print("Expected Result: Value should approximate these:")
    print(
        "Loss: {}\nGradient wrt Center Vectors(dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            outputs['test_naivesoftmax']['loss'],
            outputs['test_naivesoftmax']['dj_dvc'],
            outputs['test_naivesoftmax']['dj_du']))
    return (outputs['test_naivesoftmax']['loss'], outputs['test_naivesoftmax']['dj_dvc'], outputs['test_naivesoftmax']['dj_du']), (loss, dj_dvc, dj_du)

def test_sigmoid():
    print("\t\t\ttest sigmoid\t\t\t")

    x = inputs['test_sigmoid']['x']
    s = model.sigmoid(x)

    print("\nYour Result:")
    print(s)
    print("Expected Result: Value should approximate these:")
    print(outputs['test_sigmoid']['s'])
    return outputs['test_sigmoid']['s'], s

def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset, dummy_vectors, dummy_tokens = dummy()

    print("==== Gradient check for skip-gram with naive_softmax_loss_and_gradient ====")
    gradcheck_passed = gradcheck_naive(lambda vec: model.word2vec_sgd_wrapper(
        model.skipgram, dummy_tokens, vec, dataset, 5, model.naive_softmax_loss_and_gradient),
                    dummy_vectors, "naive_softmax_loss_and_gradient Gradient")

    print("\n\t\t\tSkip-Gram with naive_softmax_loss_and_gradient\t\t\t")

    print("\nYour Result:")
    loss, dj_dv, dj_du = model.skipgram(inputs['test_word2vec']['current_center_word'], inputs['test_word2vec']['window_size'],
                                  inputs['test_word2vec']['outside_words'],
                                  dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
                                  model.naive_softmax_loss_and_gradient)
    print(
        "Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(loss,
                                                                                                                   dj_dv,
                                                                                                                   dj_du))

    print("Expected Result: Value should approximate these:")
    print(
        "Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            outputs['test_word2vec']['loss'],
            outputs['test_word2vec']['dj_dv'],
            outputs['test_word2vec']['dj_du']))
    return gradcheck_passed, (outputs['test_word2vec']['loss'], outputs['test_word2vec']['dj_dv'], outputs['test_word2vec']['dj_du']), (loss, dj_dv, dj_du)

def hidden_test_sigmoid(x, soln, subm):
    points = 0
    model_result = subm.sigmoid(x.copy())

    soln_result = soln.sigmoid(x.copy())

    return np.allclose(model_result, soln_result, atol=1e-6)

def hidden_test_normalize_rows(x, soln, subm):
    return np.allclose(subm.normalize_rows(x.copy()),
                         soln.normalize_rows(x.copy()), atol=1e-4)

def hidden_dummy(soln, subm):
    random.seed(31415)
    np.random.seed(9265)

    dataset = type('dummy', (), {})()
    tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]

    def dummy_sample_token_idx():
        return random.randint(0, len(tokens) - 1)

    def get_random_context(C):
        return tokens[random.randint(0, len(tokens) - 1)], [tokens[random.randint(0, len(tokens) - 1)] \
                                                            for i in xrange(2 * C)]

    dataset.sample_token_idx = dummy_sample_token_idx
    dataset.get_random_context = get_random_context

    # random.seed(31415)
    # np.random.seed(9265)

    random.seed(35436)
    np.random.seed(4355)

    # use the normalize rows
    dummy_vectors = normalize_rows(np.random.randn(2 * len(tokens), 3))
    dummy_tokens = {}
    for i, token in enumerate(tokens):
        dummy_tokens[token] = i
    window = ["a", "b", "e", "d", "b", "c"]

    return dataset, dummy_vectors, dummy_tokens, tokens, window

def hidden_test_softmax(soln, subm):
    random.seed(34634)
    np.random.seed(2435)

    dataset, dummy_vectors, dummy_tokens, tokens, window = hidden_dummy(soln, subm)

    # vec size is 10
    center_word_vec = np.random.rand(10)
    outside_word_idx = 2

    # 15 words in the vocabulary
    outside_vectors = np.random.rand(15, 10)

    model_loss, model_grad_center_vec, model_grad_outside_vecs = subm.naive_softmax_loss_and_gradient(
        center_word_vec,
        outside_word_idx,
        outside_vectors,
        dataset)

    soln_loss, soln_grad_center_vec, soln_grad_outside_vecs = soln.naive_softmax_loss_and_gradient(
        center_word_vec,
        outside_word_idx,
        outside_vectors,
        dataset)

    return (np.isclose(model_loss, soln_loss, atol=1e-5), # cost does not match
            np.allclose(model_grad_center_vec, soln_grad_center_vec, atol=1e-5), # grad dj/v_c does not match
            np.allclose(model_grad_outside_vecs, soln_grad_outside_vecs, atol=1e-5) # grad dj/dU does not match
            )

def hidden_test_word2vec(soln, subm):
    model2 = subm.skipgram
    soln_model, soln_cost_and_grad = soln.skipgram, soln.naive_softmax_loss_and_gradient

    random.seed(34634)
    np.random.seed(2435)

    dataset, dummy_vectors, dummy_tokens, tokens, window = hidden_dummy(soln, subm)

    model_cost, model_grad_pred, model_grad = model2(
        "c", 3, window,
        dummy_tokens, dummy_vectors[:len(tokens), :], dummy_vectors[len(tokens):, :], dataset,
        soln_cost_and_grad)

    soln_cost, soln_grad_pred, soln_grad = soln_model(
        "c", 3, window,
        dummy_tokens, dummy_vectors[:len(tokens), :], dummy_vectors[len(tokens):, :], dataset,
        soln_cost_and_grad)

    return (np.isclose(model_cost, soln_cost, atol=1e-5), # cost does not match
            np.allclose(model_grad_pred, soln_grad_pred, atol=1e-5), # grad v_hat does not match
            np.allclose(model_grad, soln_grad, atol=1e-5) # grad vectors does not match
            )

def test_sgd(f, x, optimal):
  return np.allclose(
    model.sgd(f, x, 0.01, 1000, PRINT_EVERY=10000),
    optimal, atol=1e-6)

def cosine_sim(vec1, vec2):
  distance = spatial.distance.cosine(vec1, vec2)
  return 1 - distance

#########
# TESTS #
#########
class Test_2a(GradedTestCase):
  def setUp(self):
    np.random.seed(224)

  @graded()
  def test_0(self):
    """2a-0-basic:  Word2vec sanity check 1"""
    passed, sol, sub = test_word2vec()
    self.assertTrue(passed)
    self.assertTrue(np.allclose(sol[0], sub[0], rtol=1e-3))
    self.assertTrue(np.allclose(sol[1], sub[1], rtol=1e-3))
    self.assertTrue(np.allclose(sol[2], sub[2], rtol=1e-3))

  @graded()
  def test_1(self):
    """2a-1-basic:  Word2vec sanity check 2"""
    sol, sub = test_naive_softmax_loss_and_gradient()
    self.assertTrue(np.allclose(sol[0], sub[0], rtol=1e-3))
    self.assertTrue(np.allclose(sol[1], sub[1], rtol=1e-3))
    self.assertTrue(np.allclose(sol[2], sub[2], rtol=1e-3))

  @graded()
  def test_2(self):
    """2a-2-basic:  Word2vec sanity check 3"""
    sol, sub = test_sigmoid()
    self.assertTrue(np.allclose(sol, sub, rtol=1e-3))

  @graded(is_hidden=True, timeout=15)
  def test_3(self):
    """2a-3-hidden:  Sigmoid with 1D inputs"""
    self.assertTrue(hidden_test_sigmoid(np.array([-100, -3 - 2, -1, 0, 1, 2, 3, 100]),
                                        self.run_with_solution_if_possible(model, lambda sub_or_sol:sub_or_sol), model))

  @graded(is_hidden=True, timeout=15)
  def test_4(self):
    """2a-4-hidden: Sigmoid with 2D inputs"""
    self.assertTrue(hidden_test_sigmoid(np.array([[1, 2], [-1, -2]]),
                                        self.run_with_solution_if_possible(model, lambda sub_or_sol:sub_or_sol), model))

  @graded(is_hidden=True, timeout=15)
  def test_5(self):
    """2a-5-hidden:  Sigmoid with large 2D inputs"""
    self.assertTrue(hidden_test_sigmoid(np.random.rand(1000, 1000),
                                        self.run_with_solution_if_possible(model, lambda sub_or_sol:sub_or_sol), model))

  @graded(is_hidden=True, timeout=10)
  def test_6(self):
    """2a-6-hidden:  test softmax"""
    test_cost, test_grad_v_c, test_grad_U = hidden_test_softmax(self.run_with_solution_if_possible(model, lambda sub_or_sol:sub_or_sol), model)
    self.assertTrue(test_cost and test_grad_U and test_grad_v_c)

  @graded(is_hidden=True, timeout=10)
  def test_7(self):
    """2a-7-hidden:  test skipgram"""
    test_cost, test_grad_v_c, test_grad_U = hidden_test_word2vec(self.run_with_solution_if_possible(model, lambda sub_or_sol:sub_or_sol), model)
    self.assertTrue(test_cost and test_grad_U and test_grad_v_c)

class Test_2b(GradedTestCase):
  def setUp(self):
    self.quad = lambda x: (np.sum(x ** 2), x * 2)

  @graded()
  def test_0(self):
    """2b-0-basic:  SGD sanity check 1"""
    t1 = model.sgd(self.quad, 0.5, 0.01, 1000, PRINT_EVERY=100)
    print("test 1 result:", t1)
    self.assertLessEqual(abs(t1), 1e-6)

  @graded()
  def test_1(self):
    """2b-1-basic:  SGD sanity check 2"""
    t2 = model.sgd(self.quad, 0.0, 0.01, 1000, PRINT_EVERY=100)
    print("test 2 result:", t2)
    self.assertLessEqual(abs(t2), 1e-6)

  @graded()
  def test_2(self):
    """2b-2-basic:  SGD sanity check 3"""
    t3 = model.sgd(self.quad, -1.5, 0.01, 1000, PRINT_EVERY=100)
    print("test 3 result:", t3)
    self.assertLessEqual(abs(t3), 1e-6)

  @graded(is_hidden=True)
  def test_3(self):
    """2b-3-hidden:  sgd quad scalar"""
    quad = lambda x: (np.sum(x ** 2), x * 2)
    self.assertTrue(test_sgd(quad, 1, 0))

  @graded(is_hidden=True)
  def test_4(self):
    """2b-4-hidden:  sgd quad matrix"""
    quad = lambda x: (np.sum(x ** 2), x * 2)
    self.assertTrue(test_sgd(quad, (np.random.rand(100, 100) - 0.5) * 2, np.zeros((100, 100))))

class Test_2c(GradedTestCase):
  @graded()
  def test_0(self):
    """2c-0-basic:  Sanity check for word2vec implementation"""
    
    # tester code for checking the output vectors generated by run.py.
    sample_vectors_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sample_vectors_(soln).json")
    if not os.path.isfile(sample_vectors_path):
      raise Exception('Excecute run.py to generate sample_vectors_(soln).json')

    sample_vectors_actual = load(sample_vectors_path)

    test_words = ["female", "cool"]

    for word in test_words:
      print("Your output:")
      print(sample_vectors_actual[word])
      print("Expected output")
      print(sample_vectors_expected[word])
      print()
      self.assertTrue(np.allclose(sample_vectors_actual[word], sample_vectors_expected[word], rtol=1e-3))

  @graded(is_hidden=True, timeout=300)
  def test_1(self):
    """2c-1-hidden:  Compare word vector outputs (sample_vectors_soln.json) with solution."""
    self.assertTrue(os.path.exists('sample_vectors_(soln).json'), 'Cannot run unit test because word vector file is not present.  It must be uploaded with your model.py file.  Execute src/run.py to create the word vector file (sample_vectors_soln.json).')
    soln_vectors = load('sample_vectors_(soln)_compare.json')
    model_vectors = load('sample_vectors_(soln).json')
    keys = soln_vectors.keys()
    n_vectors = len(keys)

    for key in keys:
      soln_vector = soln_vectors[key]
      model_vector = model_vectors.get(key, [0] * len(soln_vector))

      self.assertTrue(np.allclose(model_vector, soln_vector, atol=1e-3))

def getTestCaseForTestID(test_id):
  question, part, _ = test_id.split('-')
  g = globals().copy()
  for name, obj in g.items():
    if inspect.isclass(obj) and name == ('Test_'+question):
      return obj('test_'+part)

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument('test_case', nargs='?', default='all')
  test_id = parser.parse_args().test_case

  assignment = unittest.TestSuite()
  if test_id != 'all':
    assignment.addTest(getTestCaseForTestID(test_id))
  else:
    assignment.addTests(unittest.defaultTestLoader.discover('.', pattern='tester.py'))
  CourseTestRunner().run(assignment)
