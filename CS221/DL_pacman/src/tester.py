#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
from testerUtil import graded, CourseTestRunner, GradedTestCase
from game import Agent
from ghostAgents import RandomGhost, DirectionalGhost
import pacman, time, layout, textDisplay
textDisplay.SLEEP_TIME = 0
textDisplay.DRAW_EVERY = 1000

import runmodel

# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
SEED = 'testing' # random seed at the beginning of each question for more fairness in grading...

def run(layname, pac, ghosts, nGames = 1, name = 'games'):
  """
  Runs a few games and outputs statistics.
  """

  starttime = time.time()
  lay = layout.getLayout(layname, 3)
  disp = textDisplay.NullGraphics()

  print(('*** Running %s on' % name, layname,'%d time(s).' % nGames))
  games = pacman.runGames(lay, pac, ghosts, disp, nGames, False, catchExceptions=False)
  print(('*** Finished running %s on' % name, layname,'after %d seconds.' % (time.time() - starttime)))

  stats = {'time': time.time() - starttime, 'wins': [g.state.isWin() for g in games].count(True), 'games': games, 'scores': [g.state.getScore() for g in games], 'timeouts': [g.agentTimeout for g in games].count(True)}
  print(('*** Won %d out of %d games. Average score: %f ***' % (stats['wins'], len(games), sum(stats['scores']) * 1.0 / len(games))))

  return stats

def comparison_checking(theirPac, ourPacOptions, agentName):
  """
  Takes in the Pacman agent, wraps it in ours, and assigns points.
  """
  print('Running our tester (hidden from you)...')
  random.seed(SEED)
  offByOne = False
  partialPlyBug = False
  totalSuboptimal = 0
  timeout = False

  return timeout, offByOne, partialPlyBug, totalSuboptimal

def runq4():
  """
  Runs the expectimax agent a few times and checks for victory!
  """
  random.seed(SEED)
  nGames = 20

  print(('Running your agent %d times to compute the average score...' % nGames))
  print(('The timeout message (if any) is obtained by running the game once, rather than %d times' % nGames))
  params = '-l smallClassic -p ExpectimaxAgent -a evalFn=better -q -n %d -c' % nGames
  games = pacman.runGames(**pacman.readCommand(params.split(' ')))
  timeouts = [game.agentTimeout for game in games].count(True)
  wins = [game.state.isWin() for game in games].count(True)
  averageWinScore = 0
  if wins >= nGames / 2:
    scores = [game.state.getScore() for game in games if game.state.isWin()]
    averageWinScore = sum(scores) / len(scores)
  print(('Average score of winning games: %d \n' % averageWinScore))
  return timeouts, wins, averageWinScore

class TestCase_A4(GradedTestCase):
  def setUp(self):
    self.gamePlay = {}

  def timeout_test(self, agentName):
    stats = {}
    if agentName == 'alphabeta':
      stats = run('smallClassic', runmodel.AlphaBetaAgent(depth=2), [DirectionalGhost(i + 1) for i in range(2)], name='%s (depth %d)' % ('alphabeta', 2))
    elif agentName == 'minimax':
      stats = run('smallClassic', runmodel.MinimaxAgent(depth=2), [DirectionalGhost(i + 1) for i in range(2)], name='%s (depth %d)' % ('minimax', 2))
    else:
      stats = run('smallClassic', runmodel.ExpectimaxAgent(depth=2), [DirectionalGhost(i + 1) for i in range(2)], name='%s (depth %d)' % ('expectimax', 2))
    print(agentName)
    print(stats['timeouts'])
    self.assertLessEqual(stats['timeouts'], 0,
                       msg=f'Your {agentName} agent timed out on smallClassic.  No autotester feedback will be provided.')

  def OBOB_test(self, agentName):
    if agentName not in self.gamePlay:
      if agentName == 'minimax':
        self.gamePlay[agentName] = comparison_checking(runmodel.MinimaxAgent(depth=2), {}, agentName)
      elif agentName == 'alphabeta':
        self.gamePlay[agentName] =  comparison_checking(runmodel.AlphaBetaAgent(depth=2), {agentName: 'True'}, agentName)
      elif agentName == 'expectimax':
        self.gamePlay[agentName] = comparison_checking(runmodel.ExpectimaxAgent(depth=2), {agentName: 'True'}, agentName)
      else:
        raise Exception("Unexpected agent name: " + agentName)

    timeout, offByOne, partialPlyBug, totalSuboptimal = self.gamePlay[agentName]
    self.assertFalse(timeout, msg=f'Your {agentName} agent timed out on smallClassic.  No autotester feedback will be provided.')
    self.assertFalse(offByOne, 'Depth off by 1')

  def search_depth_test(self, agentName):
    if agentName not in self.gamePlay:
      if agentName == 'minimax':
        self.gamePlay[agentName] = comparison_checking(runmodel.MinimaxAgent(depth=2), {}, agentName)
      elif agentName == 'alphabeta':
        self.gamePlay[agentName] =  comparison_checking(runmodel.AlphaBetaAgent(depth=2), {agentName: 'True'}, agentName)
      elif agentName == 'expectimax':
        self.gamePlay[agentName] = comparison_checking(runmodel.ExpectimaxAgent(depth=2), {agentName: 'True'}, agentName)
      else:
        raise Exception("Unexpected agent name: " + agentName)

    timeout, offByOne, partialPlyBug, totalSuboptimal = self.gamePlay[agentName]
    self.assertFalse(timeout, msg=f'Your {agentName} agent timed out on smallClassic.  No autotester feedback will be provided.')
    self.assertFalse(partialPlyBug, msg='Incomplete final search ply bug')

  def suboptimal_test(self, agentName):
    if agentName not in self.gamePlay:
      if agentName == 'minimax':
        self.gamePlay[agentName] = comparison_checking(runmodel.MinimaxAgent(depth=2), {}, agentName)
      elif agentName == 'alphabeta':
        self.gamePlay[agentName] =  comparison_checking(runmodel.AlphaBetaAgent(depth=2), {agentName: 'True'}, agentName)
      elif agentName == 'expectimax':
        self.gamePlay[agentName] = comparison_checking(runmodel.ExpectimaxAgent(depth=2), {agentName: 'True'}, agentName)
      else:
        raise Exception("Unexpected agent name: " + agentName)

    timeout, offByOne, partialPlyBug, totalSuboptimal = self.gamePlay[agentName]
    self.assertFalse(timeout, msg=f'Your {agentName} agent timed out on smallClassic.  No autotester feedback will be provided.')
    self.assertLessEqual(totalSuboptimal, 0, msg=f'Suboptimal moves: {totalSuboptimal}')

# TESTS #
class Test_1b(TestCase_A4):
  @graded(timeout=10)
  def test_0(self):
    """1b-0-basic:  Tests minimax for timeout on smallClassic."""
    self.timeout_test('minimax')

  @graded(timeout=10, is_hidden=True)
  def test_1(self):
    """1b-1-hidden:  Tests minimax for off by one bug on smallClassic."""
    self.OBOB_test('minimax')

  @graded(timeout=10, is_hidden=True)
  def test_2(self):
    """1b-2-hidden:  Tests minimax for search depth bug on smallClassic."""
    self.search_depth_test('minimax')

  @graded(timeout=10, is_hidden=True)
  def test_3(self):
    """1b-3-hidden:  Tests minimax for suboptimal moves on smallClassic."""
    self.suboptimal_test('minimax')

class Test_2a(TestCase_A4):
  @graded(timeout=10)
  def test_0(self):
    """2a-0-basic:  Tests alphabeta for timeout on smallClassic."""
    self.timeout_test('alphabeta')

  @graded(timeout=10, is_hidden=True)
  def test_1(self):
    """2a-1-hidden:  Tests alphabeta for off by one bug on smallClassic."""
    self.OBOB_test('alphabeta')

  @graded(timeout=10, is_hidden=True)
  def test_2(self):
    """2a-2-hidden:  Tests alphabeta for search depth bug on smallClassic."""
    self.search_depth_test('alphabeta')

  @graded(timeout=10, is_hidden=True)
  def test_3(self):
    """2a-3-hidden:  Tests alphabeta for suboptimal moves on smallClassic."""
    self.suboptimal_test('alphabeta')

class Test_3b(TestCase_A4):
  @graded(timeout=10)
  def test_0(self):
    """3b-0-basic:  Tests expectimax for timeout on smallClassic."""
    self.timeout_test('expectimax')

  @graded(timeout=10, is_hidden=True)
  def test_1(self):
    """3b-1-hidden:  Tests expectimax for off by one bug on smallClassic."""
    self.OBOB_test('expectimax')

  @graded(timeout=10, is_hidden=True)
  def test_2(self):
    """3b-2-hidden:  Tests expectimax for search depth bug on smallClassic."""
    self.search_depth_test('expectimax')

  @graded(timeout=10, is_hidden=True)
  def test_3(self):
    """3b-3-hidden:  Tests expectimax for suboptimal moves on smallClassic."""
    self.suboptimal_test('expectimax')

class Test_4a(TestCase_A4):
  @graded(timeout=10,
          is_extra_credit=True,
          leaderboard_col_name='Average Score (20 games)')
  def test_0(self, set_leaderboard_value):
    """4a-0-basic:  1 extra credit point per 100 point increase above 1300."""
    timeouts, wins, averageWinScore = runq4()

    self.assertLessEqual(timeouts, 0, msg='Agent timed out on smallClassic with betterEvaluationFunction. No autotester feedback will be provided.')
    self.assertGreater(wins, 0, msg='Your better evaluation function never won any games.')

    extra_credit = min((averageWinScore - 1300) // 100, 8)
    
    self.assertGreater(extra_credit, 0, msg=f'A valid extra credit should be higher than 0. Extra credit received: {extra_credit}')

    self.earned = int(extra_credit)
    set_leaderboard_value(averageWinScore)

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