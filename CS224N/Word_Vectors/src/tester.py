#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
from testerUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np
import traceback

import run
from utils import *

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
#############################################

def toy_corpus():
  toy_corpus = ["START All that glitters isn't gold END".split(" "), "START All's well that ends well END".split(" ")]
  return toy_corpus

def toy_corpus_co_occurrence():
  ### co-occurrence matric for toy_corpus with window_size = 2 
  M = np.array(
  [[0., 0., 0., 1., 0., 1., 0., 0., 1., 0.,],
   [0., 0., 0., 1., 0., 0., 0., 0., 1., 1.,],
   [0., 0., 0., 0., 1., 0., 1., 1., 0., 1.,],
   [1., 1., 0., 0., 0., 0., 0., 0., 1., 1.,],
   [0., 0., 1., 0., 0., 0., 0., 0., 1., 2.,],
   [1., 0., 0., 0., 0., 0., 1., 1., 1., 0.,],
   [0., 0., 1., 0., 0., 1., 0., 1., 0., 0.,],
   [0., 0., 1., 0., 0., 1., 1., 0., 1., 0.,],
   [1., 1., 0., 1., 1., 1., 0., 1., 0., 2.,],
   [0., 1., 1., 1., 2., 0., 0., 0., 2., 0.,]]
  )

  word2Ind = {'All': 0, "All's": 1, 'END': 2, 'START': 3, 'ends': 4, 'glitters': 5, 'gold': 6, "isn't": 7, 'that': 8, 'well': 9}
  return M, word2Ind

#########
# TESTS #
#########

class Test_1(GradedTestCase):
  def setUp(self):
    np.random.seed(42)

  @graded()
  def test_0(self):
    """1-0-basic:  Sanity check for distinct_words()"""

    test_corpus = toy_corpus()
    test_corpus_words, num_corpus_words = run.distinct_words(test_corpus)

    ans_test_corpus_words = sorted(list(set(["START", "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", "END"])))
    ans_num_corpus_words = len(ans_test_corpus_words)

    self.assertEqual(test_corpus_words, ans_test_corpus_words)
    self.assertEqual(num_corpus_words, ans_num_corpus_words)

  @graded()
  def test_1(self):
    """1-1-basic:  Sanity check for compute_co_occurrence_matrix()"""

    test_corpus = toy_corpus()
    M_test, word2Ind_test = run.compute_co_occurrence_matrix(test_corpus, window_size=2)

    M_test_ans, word2Ind_test_ans = toy_corpus_co_occurrence()

    for w1 in word2Ind_test_ans.keys():
        idx1 = word2Ind_test_ans[w1]
        for w2 in word2Ind_test_ans.keys():
            idx2 = word2Ind_test_ans[w2]
            model = M_test[idx1, idx2]
            correct = M_test_ans[idx1, idx2]
            if model != correct:
                print("Correct M:")
                print(M_test_ans)
                print("Your M: ")
                print(M_test)
                self.assertEqual(model, correct, "Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(idx1, idx2, w1, w2, model, correct))
    self.assertSequenceEqual(M_test.shape, M_test_ans.shape)
    self.assertSequenceEqual(word2Ind_test, word2Ind_test_ans)

  @graded()
  def test_2(self):
    """1-2-basic:  Sanity check for reduce_to_k_dim()"""

    M_test_ans, word2Ind_test_ans = toy_corpus_co_occurrence()
    M_test_reduced = run.reduce_to_k_dim(M_test_ans, k=2)
    self.assertSequenceEqual(M_test_reduced.shape, (10,2))

  @graded(is_hidden=True)
  def test_3(self):
    """1-3-hidden:  Test distinct_words() with full corpus."""
    corpus = read_corpus()

    model_result, _ = run.distinct_words(corpus.copy())
    soln_result, _ = self.run_with_solution_if_possible(run, lambda sub_or_sol:sub_or_sol.distinct_words(corpus.copy()))

    self.assertEqual(model_result, soln_result)

  @graded(is_hidden=True, timeout=20)
  def test_4(self):
    """1-4-hidden:  Test compute_co_occurrence_matrix() with full corpus."""
    corpus = read_corpus()
    window_size = 4
    model_matrix, model_dict = run.compute_co_occurrence_matrix(corpus.copy(), window_size)
    soln_matrix, solution_dict = self.run_with_solution_if_possible(run, lambda sub_or_sol:sub_or_sol.compute_co_occurrence_matrix(corpus.copy(), window_size))

    self.assertEqual(np.linalg.norm(model_matrix - soln_matrix), 0)
    self.assertEqual(solution_dict, model_dict)

  @graded(is_hidden=True)
  def test_5(self):
    """1-5-hidden:  Test reduce_to_k_dim() with full corpus."""
    random.seed(35436)
    np.random.seed(4355)

    x = 10*np.random.rand(50, 100) + 100
    k = 5

    model_result = run.reduce_to_k_dim(x.copy(), k)

    soln_result = self.run_with_solution_if_possible(run, lambda sub_or_sol: sub_or_sol.reduce_to_k_dim(x.copy(), k))

    self.assertTrue(np.allclose(model_result, soln_result, atol=1e-5))

def getTestCaseForTestID(test_id):
  question, part, _ = test_id.split('-')
  g = globals().copy()
  for name, obj in g.items():
    if inspect.isclass(obj) and name == ('Test_'+question):
      return obj('test_'+part)

if __name__ == '__main__':

  # Parse for a specific test case
  parser = argparse.ArgumentParser()
  parser.add_argument('test_case', nargs='?', default='all')
  test_id = parser.parse_args().test_case

  assignment = unittest.TestSuite()
  if test_id != 'all':
    assignment.addTest(getTestCaseForTestID(test_id))
  else:
    assignment.addTests(unittest.defaultTestLoader.discover('.', pattern='tester.py'))
  CourseTestRunner().run(assignment)
