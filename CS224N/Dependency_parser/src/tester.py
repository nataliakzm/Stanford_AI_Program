#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
from testerUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np
import os
import traceback

import run_model

from run_model.parser_utils import minibatches, load_and_preprocess_data
import torch
from datetime import datetime

# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
def test_step(name, transition, stack, buf, deps,ex_stack, ex_buf, ex_deps):
    """Tests that a single parse step returns the expected output"""
    pp = run_model.PartialParse([])
    pp.stack, pp.buffer, pp.dependencies = stack, buf, deps

    pp.parse_step(transition)
    stack, buf, deps = (tuple(pp.stack), tuple(pp.buffer), tuple(sorted(pp.dependencies)))
    assert stack == ex_stack, \
        "{:} test resulted in stack {:}, expected {:}".format(name, stack, ex_stack)
    assert buf == ex_buf, \
        "{:} test resulted in buffer {:}, expected {:}".format(name, buf, ex_buf)
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)
    print("{:} test passed!".format(name))

def test_parse_step():
    """Simple tests for the PartialParse.parse_step function
    Warning: these are not exhaustive
    """
    test_step("SHIFT", "S", ["ROOT", "the"], ["cat", "sat"], [],
              ("ROOT", "the", "cat"), ("sat",), ())
    test_step("LEFT-ARC", "LA", ["ROOT", "the", "cat"], ["sat"], [],
              ("ROOT", "cat",), ("sat",), (("cat", "the"),))
    test_step("RIGHT-ARC", "RA", ["ROOT", "run", "fast"], [], [],
              ("ROOT", "run",), (), (("run", "fast"),))

def test_parse():
    """Simple tests for the PartialParse.parse function
    Warning: these are not exhaustive
    """
    sentence = ["parse", "this", "sentence"]
    dependencies = run_model.PartialParse(sentence).parse(["S", "S", "S", "LA", "RA", "RA"])
    dependencies = tuple(sorted(dependencies))
    expected = (('ROOT', 'parse'), ('parse', 'sentence'), ('sentence', 'this'))
    assert dependencies == expected, \
        "parse test resulted in dependencies {:}, expected {:}".format(dependencies, expected)
    assert tuple(sentence) == ("parse", "this", "sentence"), \
        "parse test failed: the input sentence should not be modified"
    print("parse test passed!")

class DummyModel(object):
    """Dummy model for testing the minibatch_parse function
    First shifts everything onto the stack and then does exclusively right arcs if the first word of
    the sentence is "right", "left" if otherwise.
    """

    def predict(self, partial_parses):
        return [("RA" if pp.stack[1] == "right" else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]

def test_dependencies(name, deps, ex_deps):
    """Tests the provided dependencies match the expected dependencies"""
    deps = tuple(sorted(deps))
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)

def test_minibatch_parse():
    """Simple tests for the minibatch_parse function
    Warning: these are not exhaustive
    """
    sentences = [["right", "arcs", "only"],
                 ["right", "arcs", "only", "again"],
                 ["left", "arcs", "only"],
                 ["left", "arcs", "only", "again"]]
    deps = run_model.minibatch_parse(sentences, DummyModel(), 2)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'right'), ('arcs', 'only'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[1],
                      (('ROOT', 'right'), ('arcs', 'only'), ('only', 'again'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[2],
                      (('only', 'ROOT'), ('only', 'arcs'), ('only', 'left')))
    test_dependencies("minibatch_parse", deps[3],
                      (('again', 'ROOT'), ('again', 'arcs'), ('again', 'left'), ('again', 'only')))
    print("minibatch_parse test passed!")


def parses_equal(p1, p2):
    return tuple(p1.stack) == tuple(p2.stack) and \
           tuple(p1.buffer) == tuple(p2.buffer) and \
           tuple(sorted(p1.dependencies)) == tuple(sorted(p2.dependencies))

def test_init(soln):
    sentence = [0, 1, 2, 3]
    return parses_equal(soln.PartialParse(sentence), run_model.PartialParse(sentence))

def run_parse_step(PartialParse, transition, stack, buf, deps):
    pp = PartialParse([])
    pp.stack, pp.buffer, pp.dependencies = stack[:], buf[:], deps[:]
    pp.parse_step(transition)
    return pp

def hidden_test_parse_step(transition, stack, buf, deps, soln):
    expected = run_parse_step(soln.PartialParse, transition, stack, buf, deps)
    model_result = run_parse_step(run_model.PartialParse, transition, stack, buf, deps)
    return parses_equal(expected, model_result)

def hidden_test_parse(soln):
    sentence = [1, 2, 3]
    transitions = ["S", "S", "S", "LA", "RA", "RA"]
    expected = soln.PartialParse(sentence).parse(transitions)
    model_result = run_model.PartialParse(sentence).parse(transitions)
    return expected == model_result

def hidden_test_minibatch_parse(sentences, batch_size, soln):
    expected = soln.minibatch_parse(sentences, soln.parser_transitions.DummyModel(), batch_size)
    actual = run_model.minibatch_parse(sentences, run_model.parser_transitions.DummyModel(), batch_size)
    return len(expected) == len(actual) and all(
        tuple(sorted(expected[i])) == tuple(sorted(actual[i])) for i in range(len(expected)))

def hidden_test_parser_model(input, soln):

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True

    _, embeddings, _, _, _ = run_model.load_and_preprocess_data()
    
    model_expected = soln.ParserModel(embeddings).forward(input).data.numpy().tolist()
    model_actual = run_model.ParserModel(embeddings).forward(input).data.numpy().tolist()

    print("actual output")
    print(model_actual)
    print()

    print("expected output")
    print(model_expected)

    return np.isclose(model_actual, model_expected, atol=1e-2).all()

def setup():
    # IMP need to change the data format here
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(True)
    np.random.seed(0)
    model = run_model.ParserModel(embeddings)
    parser.model = model
    return parser, embeddings, train_data, dev_data, test_data

def test_predict(parser, embeddings, train_data, dev_data, test_data, batch_size=2048):
    parser.model.eval()
    with torch.no_grad():
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            train_x = torch.from_numpy(train_x).long()
            train_y = torch.from_numpy(train_y.nonzero()[1]).long()
            results = parser.model(train_x)
            return np.shape(results.numpy()) == (batch_size, 3)

def xavier_test_tensor(weights):
    return type(weights) == torch.nn.parameter.Parameter

def xavier_test_range(weights):
    return np.min(weights.detach().numpy()) < 0 and np.max(weights.detach().numpy()) > 0

def xavier_test_bounds(weights):
    correct = True
    val = weights.detach().numpy()
    epsilon = epsilon = np.sqrt(6.0 / np.sum(list(weights.size())))
    correct &= val.max() < epsilon
    correct &= val.min() > -epsilon
    return correct

def test_xavier(weights):
    return xavier_test_bounds(weights) and xavier_test_range(weights) and xavier_test_tensor(weights)

def uses_xavier(parser, embeddings, train_data, dev_data, test_data):
    model = parser.model
    parser.model.eval()
    with torch.no_grad():
        return test_xavier(model.embed_to_hidden.weight) and test_xavier(model.hidden_to_logits.weight)

def test_parser_and_train(parser, embeddings, train_data, dev_data, test_data):
    output_dir = "autotester_(soln)/results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    run_model.train(parser, train_data, dev_data, output_path, batch_size=32, n_epochs=2, lr=0.0005)
    parser.model.eval()  # Places model in "eval" mode, i.e. don't apply dropout layer
    UAS, _ = parser.parse(test_data)
    return UAS

test_cases_ip = {
    'parser_model': {
        't': torch.tensor([[5155, 5156, 2429, 89, 2430, 2431, 101, 5155, 103, 5155, 5155, 5155,
                            5155, 5155, 5155, 5155, 5155, 5155, 83, 84, 43, 52, 50, 43,
                            54, 83, 48, 83, 83, 83, 83, 83, 83, 83, 83, 83],
                           [2936, 89, 2937, 92, 138, 1451, 5155, 5155, 5155, 5155, 5155, 5155,
                            5155, 5155, 5155, 5155, 5155, 5155, 50, 52, 54, 51, 59, 44,
                            83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83],
                           [2429, 338, 86, 250, 88, 101, 5155, 5155, 5155, 5155, 5155, 5155,
                            287, 5155, 85, 5155, 5155, 5155, 43, 39, 45, 47, 40, 54,
                            83, 83, 83, 83, 83, 83, 40, 83, 41, 83, 83, 83],
                           [5155, 5156, 2051, 144, 4412, 86, 91, 97, 96, 5155, 5155, 5155,
                            5155, 5155, 5155, 5155, 5155, 5155, 83, 84, 39, 71, 42, 45,
                            40, 62, 61, 83, 83, 83, 83, 83, 83, 83, 83, 83],
                           [5156, 571, 86, 535, 401, 92, 5155, 5155, 5155, 5155, 5155, 5155,
                            267, 1054, 5155, 5155, 5155, 1278, 84, 44, 45, 43, 39, 51,
                            83, 83, 83, 83, 83, 83, 56, 53, 83, 83, 83, 44]])
    },
    'run': 66.50004165625261
}

#########
# TESTS #
#########
class Test_1a(GradedTestCase):
  def setUp(self):
    random.seed(35436)
    np.random.seed(4355)

  @graded()
  def test_0(self):
    """1a-0-basic:  Sanity check for PartialParse.parse_step"""
    test_parse_step()

  @graded()
  def test_1(self):
    """1a-1-basic:  Sanity check for PartialParse.parse"""
    test_parse()

  @graded(is_hidden=True)
  def test_2(self):
    """1a-2-hidden:  init"""
    self.assertTrue(test_init(self.run_with_solution_if_possible(run_model, lambda sub_or_sol:sub_or_sol)))

  @graded(is_hidden=True)
  def test_3(self):
    """1a-3-hidden:  shift"""
    self.assertTrue(hidden_test_parse_step("S", [0, 1, 2], [1], [], self.run_with_solution_if_possible(run_model, lambda sub_or_sol:sub_or_sol)))

  @graded(is_hidden=True)
  def test_4(self):
    """1a-4-hidden:  right-arc"""
    self.assertTrue(hidden_test_parse_step("RA", [0, 1, 2], [1], [], self.run_with_solution_if_possible(run_model, lambda sub_or_sol:sub_or_sol)))

  @graded(is_hidden=True)
  def test_5(self):
    """1a-5-hidden:  left-arc"""
    self.assertTrue(hidden_test_parse_step("LA", [0, 1, 2], [1], [], self.run_with_solution_if_possible(run_model, lambda sub_or_sol:sub_or_sol)))

  @graded(is_hidden=True)
  def test_6(self):
    """1a-6-hidden:  parse"""
    self.assertTrue(hidden_test_parse(self.run_with_solution_if_possible(run_model, lambda sub_or_sol:sub_or_sol)))

class Test_1b(GradedTestCase):
  def setUp(self):
    random.seed(35436)
    np.random.seed(4355)
    self.sentences_simple = [["right", "arcs", "only"],
                             ["left", "arcs", "only"],
                             ["left", "arcs", "only"]]
    self.sentences = [["right", "arcs", "only"],
                      ["right", "arcs", "only", "again"],
                      ["left", "arcs", "only"],
                      ["left", "arcs", "only", "again"]]

  @graded()
  def test_0(self):
    """1b-0-basic:  Sanity check for minibatch_parse"""
    test_minibatch_parse()


  @graded(is_hidden=True)
  def test_1(self):
    """1b-1-hidden: single batch"""
    self.assertTrue(hidden_test_minibatch_parse(self.sentences_simple, 3, self.run_with_solution_if_possible(run_model, lambda sub_or_sol:sub_or_sol)))

  @graded(is_hidden=True)
  def test_2(self):
    """1b-2-hidden: batch_size = 1"""
    self.assertTrue(hidden_test_minibatch_parse(self.sentences_simple, 1, self.run_with_solution_if_possible(run_model, lambda sub_or_sol:sub_or_sol)))

  @graded(is_hidden=True)
  def test_3(self):
    """1b-3-hidden: same_lengths"""
    self.assertTrue(hidden_test_minibatch_parse(self.sentences_simple, 2, self.run_with_solution_if_possible(run_model, lambda sub_or_sol:sub_or_sol)))

  @graded(is_hidden=True)
  def test_4(self):
    """1b-4-hidden: different_lengths"""
    self.assertTrue(hidden_test_minibatch_parse(self.sentences, 2, self.run_with_solution_if_possible(run_model, lambda sub_or_sol:sub_or_sol)))

class Test_1c(GradedTestCase):
  def setUp(self):
    random.seed(35436)
    np.random.seed(4355)
    self.inputs = setup()

  @graded(is_hidden = True, timeout=30)
  def test_0(self):
    """1c-0-hidden:  Sanity check for Parser Model"""
    self.assertTrue(hidden_test_parser_model(test_cases_ip['parser_model']['t'], self.run_with_solution_if_possible(run_model, lambda sub_or_sol:sub_or_sol)))

  @graded(is_hidden = True, timeout=30)
  def test_1(self):
    """1c-1-hidden: predict_on_batch"""
    test_predict(*self.inputs)
    self.assertTrue(True)

  @graded(is_hidden = True, timeout=100)
  def test_2(self):
    """1c-2-hidden: uses_xavier"""
    result = uses_xavier(*self.inputs)
    self.assertTrue(result)

  @graded(is_hidden = True, timeout=240)
  def test_3(self):
    """1c-3-hidden: Complete training and Test Set UAS"""
    loss_UAS = test_parser_and_train(*self.inputs)
    if loss_UAS is None:
        loss_UAS = 100
    print('Final UAS on test set is', loss_UAS)
    self.assertTrue(loss_UAS < 1 and loss_UAS > 0.65)

def getTestCaseForTestID(test_id):
  question, part, _ = test_id.split('-')
  g = globals().copy()
  for name, obj in g.items():
    if inspect.isclass(obj) and name == ('Test_'+question):
      return obj('test_'+part)

if __name__ == '__main__':
  # Parse for a specific test
  parser = argparse.ArgumentParser()
  parser.add_argument('test_case', nargs='?', default='all')
  test_id = parser.parse_args().test_case

  assignment = unittest.TestSuite()
  if test_id != 'all':
    assignment.addTest(getTestCaseForTestID(test_id))
  else:
    assignment.addTests(unittest.defaultTestLoader.discover('.', pattern='tester.py'))
  CourseTestRunner().run(assignment)
