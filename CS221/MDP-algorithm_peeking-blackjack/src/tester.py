#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
from testerUtil import graded, CourseTestRunner, GradedTestCase
import util

import model

# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
class AddNoiseMDP(util.MDP):
    def __init__(self, originalMDP):
        self.originalMDP = originalMDP

    def startState(self):
        return self.originalMDP.startState()

    # Return set of actions possible from |state|.
    def actions(self, state):
        return self.originalMDP.actions(state)

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        originalSuccAndProbReward = self.originalMDP.succAndProbReward(state, action)
        newSuccAndProbReward = []
        for state, prob, reward in originalSuccAndProbReward:
            newProb = 0.5 * prob + 0.5 / len(originalSuccAndProbReward)
            newSuccAndProbReward.append((state, newProb, reward))
        return newSuccAndProbReward

    # Return set of actions possible from |state|.
    def discount(self):
        return self.originalMDP.discount()

# TESTS #
class Test_3a(GradedTestCase):
  @graded()
  def test_basic(self):
    """3a-basic-0:  Basic test for succAndProbReward() that covers several edge cases."""
    mdp1 = model.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                   threshold=10, peekCost=1)
    startState = mdp1.startState()
    preBustState = (6, None, (1, 1))
    postBustState = (11, None, None)

    mdp2 = model.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                   threshold=15, peekCost=1)
    preEmptyState = (11, None, (1,0))

    # Make sure the succAndProbReward function is implemented correctly.
    tests = [
        ([((1, None, (1, 2)), 0.5, 0), ((5, None, (2, 1)), 0.5, 0)], mdp1, startState, 'Take'),
        ([((0, 0, (2, 2)), 0.5, -1), ((0, 1, (2, 2)), 0.5, -1)], mdp1, startState, 'Peek'),
        ([((0, None, None), 1, 0)], mdp1, startState, 'Quit'),
        ([((7, None, (0, 1)), 0.5, 0), ((11, None, None), 0.5, 0)], mdp1, preBustState, 'Take'),
        ([], mdp1, postBustState, 'Take'),
        ([], mdp1, postBustState, 'Peek'),
        ([], mdp1, postBustState, 'Quit'),
        ([((12, None, None), 1, 12)], mdp2, preEmptyState, 'Take')
    ]
    for gold, mdp, state, action in tests:
      # Uncomment this lines if you'd like to print out states/actions
      # print(('   state: {}, action: {}'.format(state, action)))
      self.assertEqual(gold, mdp.succAndProbReward(state, action))

  @graded(is_hidden=True)
  def test_hidden(self):
    """3a-hidden-0:  Hidden test for ValueIteration. Run ValueIteration on BlackjackMDP, then test if V[startState] is correct."""
    mdp = model.BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3,
                                  threshold=40, peekCost=1)
    startState = mdp.startState()
    alg = util.ValueIteration()
    alg.solve(mdp, .0001)

class Test_4a(GradedTestCase):

  @graded(timeout=10)
  def test_basic(self):
    """4a-basic-0:  Basic test for incorporateFeedback() using NumberLineMDP."""
    mdp = util.NumberLineMDP()
    mdp.computeStates()
    rl = model.QLearningAlgorithm(mdp.actions, mdp.discount(),
                                       model.identityFeatureExtractor,
                                       0)
    # We call this here so that the stepSize will be 1
    rl.numIters = 1

    rl.incorporateFeedback(0, 1, 0, 1)
    self.assertEqual(0, rl.getQ(0, -1))
    self.assertEqual(0, rl.getQ(0, 1))

    rl.incorporateFeedback(1, 1, 1, 2)
    self.assertEqual(0, rl.getQ(0, -1))
    self.assertEqual(0, rl.getQ(0, 1))
    self.assertEqual(0, rl.getQ(1, -1))
    self.assertEqual(1, rl.getQ(1, 1))

    rl.incorporateFeedback(2, -1, 1, 1)
    self.assertEqual(1.9, rl.getQ(2, -1))
    self.assertEqual(0, rl.getQ(2, 1))

  @graded(timeout=3, is_hidden=True)
  def test_hidden(self):
    """4a-hidden-0:  Hidden test for incorporateFeedback(). Run QLearningAlgorithm on smallMDP, then ensure that getQ returns reasonable value."""
    smallMDP = self.run_with_solution_if_possible(model,
                                                  lambda sub_or_sol: sub_or_sol.BlackjackMDP(cardValues=[1,5], multiplicity=2, threshold=10, peekCost=1))
    smallMDP.computeStates()
    rl = model.QLearningAlgorithm(smallMDP.actions, smallMDP.discount(),
                                   model.identityFeatureExtractor,
                                   0.2)
    util.simulate(smallMDP, rl, 30000)

class Test_4b(GradedTestCase):

  @graded(timeout=60)
  def test_helper(self):
    """4b-helper-0:  Helper function to run Q-learning simulations for question 4b."""
    model.simulate_QL_over_MDP(model.smallMDP, model.identityFeatureExtractor)
    model.simulate_QL_over_MDP(model.largeMDP, model.identityFeatureExtractor)

    self.skipTest("This test case is a helper function.")

class Test_4c(GradedTestCase):
  @graded(timeout=10)
  def test_basic(self):
    """4c-basic-0:  Basic test for blackjackFeatureExtractor.  Runs QLearningAlgorithm using blackjackFeatureExtractor, then checks to see that Q-values are correct."""
    mdp = model.BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                                  threshold=10, peekCost=1)
    mdp.computeStates()
    rl = model.QLearningAlgorithm(mdp.actions, mdp.discount(),
                                       model.blackjackFeatureExtractor,
                                       0)
    # We call this here so that the stepSize will be 1
    rl.numIters = 1

    rl.incorporateFeedback((7, None, (0, 1)), 'Quit', 7, (7, None, None))
    self.assertEqual(28, rl.getQ((7, None, (0, 1)), 'Quit'))
    self.assertEqual(7, rl.getQ((7, None, (1, 0)), 'Quit'))
    self.assertEqual(14, rl.getQ((2, None, (0, 2)), 'Quit'))
    self.assertEqual(0, rl.getQ((2, None, (0, 2)), 'Take'))

class Test_4d(GradedTestCase):
  @graded(timeout=60)
  def test_helper(self):
    """4d-helper-0:  Helper function to compare rewards when simulating RL over two different MDPs in question 4d."""
    model.compare_changed_MDP(model.originalMDP, model.newThresholdMDP, model.blackjackFeatureExtractor)
    self.skipTest("This test case is a helper function.")

class Test_5a(GradedTestCase):
  @graded(timeout=60)
  def test_helper(self):
    """5a-helper-0:  Helper function to compare optimal policies over various time horizons."""

    model.compare_MDP_Strategies(model.short_time, model.long_time)

    self.skipTest("This test case is a helper function.")

class Test_5c(GradedTestCase):
  @graded(timeout=60)
  def test_helper(self):
    """5c-helper-0:  Helper function to compare optimal policies over various discounts."""

    model.compare_MDP_Strategies(model.discounted, model.no_discount)

    self.skipTest("This test case is a helper function.")

class Test_5d(GradedTestCase):
  @graded(timeout=60)
  def test_helper(self):
    """5d-helper-0:  Helper function for exploring how optimal policies transfer across MDPs."""

    model.compare_changed_SeaLevelMDP(model.low_cost, model.high_cost)

    self.skipTest("This test case is a helper function.")


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

  project = unittest.TestSuite()
  if test_id != 'all':
    project.addTest(getTestCaseForTestID(test_id))
  else:
    project.addTests(unittest.defaultTestLoader.discover('.', pattern='tester.py'))
  CourseTestRunner().run(project)