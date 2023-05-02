#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect, collections
from testerUtil import graded, CourseTestRunner, GradedTestCase

from engine.const import Const
import util
import model

#############################################
# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
# TESTS #
#########

class Test_1a(GradedTestCase):
  @graded()
  def test_0(self):
    """1a-0-basic:  1a basic test for emission probabilities"""
    ei = model.ExactInference(10, 10)
    ei.skipElapse = True ### ONLY FOR PROBLEM 2
    ei.observe(55, 193, 200)
    self.assertAlmostEqual(0.030841805296, ei.belief.getProb(0, 0), places=4)
    self.assertAlmostEqual(0.00073380582967, ei.belief.getProb(2, 4), places=4)
    self.assertAlmostEqual(0.0269846478431, ei.belief.getProb(4, 7), places=4)
    self.assertAlmostEqual(0.0129150762582, ei.belief.getProb(5, 9), places=4)

    ei.observe(80, 250, 150)
    self.assertAlmostEqual(0.00000261584106271, ei.belief.getProb(0, 0), places=4)
    self.assertAlmostEqual(0.000924335357194, ei.belief.getProb(2, 4), places=4)
    self.assertAlmostEqual(0.0295673460685, ei.belief.getProb(4, 7), places=4)
    self.assertAlmostEqual(0.000102360275238, ei.belief.getProb(5, 9), places=4)

  @graded(is_hidden=True)
  def test_1(self):
    """1a-1-hidden:  1a test ordering of pdf"""
    oldpdf = util.pdf
    del util.pdf
    def pdf(a, b, c):
      return a + b
    util.pdf = pdf

    ei = model.ExactInference(10, 10)
    ei.skipElapse = True
    ei.observe(55, 193, 200)

    ei.observe(80, 250, 150)
    util.pdf = oldpdf # replace the old pdf

  @graded(is_hidden=True)
  def test_2(self):
    """1a-2-hidden:  1a advanced test for emission probabilities"""
    random.seed(10)

    ei = model.ExactInference(10, 10)
    ei.skipElapse = True

    N = 50
    p_values = []
    for i in range(N):
      a = int(random.random() * 300)
      b = int(random.random() * 5)
      c = int(random.random() * 300)

      ei.observe(a, b, c)

      for d in range(10):
        for e in range(10):
          p_values.append(ei.belief.getProb(d, e))

class Test_2a(GradedTestCase):
  @graded()
  def test_0(self):
    """2a-0-basic:  test correctness of elapseTime()"""
    ei = model.ExactInference(30, 13)
    ei.elapseTime()
    self.assertAlmostEqual(0.0105778989624, ei.belief.getProb(16, 6), places = 4)
    self.assertAlmostEqual(0.00250560512469, ei.belief.getProb(18, 7), places = 4)
    self.assertAlmostEqual(0.0165024135157, ei.belief.getProb(21, 7), places = 4)
    self.assertAlmostEqual(0.0178755550388, ei.belief.getProb(8, 4), places = 4)

    ei.elapseTime()
    self.assertAlmostEqual(0.0138327373012, ei.belief.getProb(16, 6), places = 4)
    self.assertAlmostEqual(0.00257237608713, ei.belief.getProb(18, 7), places = 4)
    self.assertAlmostEqual(0.0232612833688, ei.belief.getProb(21, 7), places = 4)
    self.assertAlmostEqual(0.0176501876956, ei.belief.getProb(8, 4), places = 4)

  @graded(is_hidden=True)
  def test_1i(self):
    """2a-1i-hidden:  Advanced test for transition probabilities, strict time limit."""
    A = 30
    B = 30
    random.seed(15)
    ei = model.ExactInference(A, B)

    N1 = 20
    N2 = 400
    p_values = []
    for i in range(N1):
      ei.elapseTime()
      for i in range(N2):
        d = int(random.random() * A)
        e = int(random.random() * B)
        p_values.append(ei.belief.getProb(d, e))

  @graded(timeout=20, is_hidden=True)
  def test_1ii(self):
    """2a-1ii-hidden:  2a test for transition probabilities on other maps, loose time limit"""
    random.seed(15)

    oldworld = Const.WORLD
    Const.WORLD = 'small' # well... they may have made it specific for lombard

    A = 30
    B = 30
    ei = model.ExactInference(A, B)

    N1 = 20
    N2 = 40
    p_values = []
    for i in range(N1):
      ei.elapseTime()
      for i in range(N2):
        d = int(random.random() * A)
        e = int(random.random() * B)
        p_values.append(ei.belief.getProb(d, e))
    Const.WORLD = oldworld # set it back to what's likely lombard

  @graded(is_hidden=True)
  def test_2(self):
    """2a-2-hidden:  advanced test for emission AND transition probabilities, strict time limit"""
    random.seed(20)

    A = 30
    B = 30
    ei = model.ExactInference(A, B)

    N1 = 20
    N2 = 400
    p_values = []
    for i in range(N1):
      ei.elapseTime()

      a = int(random.random() * 5 * A)
      b = int(random.random() * 5)
      c = int(random.random() * 5 * A)

      ei.observe(a, b, c)
      for i in range(N2):
        d = int(random.random() * A)
        e = int(random.random() * B)
        p_values.append(ei.belief.getProb(d, e))

class Test_3a(GradedTestCase):
  @graded()
  def test_0(self):
    """3a-0-basic:  3a basic test for PF observe"""
    random.seed(3)

    pf = model.ParticleFilter(30, 13)

    pf.observe(555, 193, 800)
    self.assertAlmostEqual(0.02, pf.belief.getProb(20, 4), places=4)
    self.assertAlmostEqual(0.04, pf.belief.getProb(21, 5), places=4)
    self.assertAlmostEqual(0.94, pf.belief.getProb(22, 6), places=4)
    self.assertAlmostEqual(0.0, pf.belief.getProb(8, 4), places=4)

    pf.observe(525, 193, 830)
    self.assertAlmostEqual(0.0, pf.belief.getProb(20, 4), places=4)
    self.assertAlmostEqual(0.0, pf.belief.getProb(21, 5), places=4)
    self.assertAlmostEqual(1.0, pf.belief.getProb(22, 6), places=4)
    self.assertAlmostEqual(0.0, pf.belief.getProb(8, 4), places=4)

  @graded()
  def test_1(self):
    """3a-1-basic:  3a basic test for PF elapseTime"""
    random.seed(3)
    pf = model.ParticleFilter(30, 13)
    self.assertAlmostEqual(69, len([k for k,v in list(pf.particles.items()) if v > 0]), places=4) # This should not fail unless your code changed the random initialization code.

    pf.elapseTime()
    self.assertAlmostEqual(200, sum(pf.particles.values()), places=4) # Do not lose particles
    self.assertAlmostEqual(58, len([k for k,v in list(pf.particles.items()) if v > 0]), places=4) # Most particles lie on the same (row, col) locations

    self.assertAlmostEqual(6, pf.particles[(3,9)], places=4)
    self.assertAlmostEqual(0, pf.particles[(2,10)], places=4)
    self.assertAlmostEqual(3, pf.particles[(8,4)], places=4)
    self.assertAlmostEqual(2, pf.particles[(12,6)], places=4)
    self.assertAlmostEqual(2, pf.particles[(7,8)], places=4)
    self.assertAlmostEqual(2, pf.particles[(11,6)], places=4)
    self.assertAlmostEqual(0, pf.particles[(18,7)], places=4)
    self.assertAlmostEqual(1, pf.particles[(20,5)], places=4)

    pf.elapseTime()
    self.assertAlmostEqual(200, sum(pf.particles.values()), places=4) # Do not lose particles
    self.assertAlmostEqual(57, len([k for k,v in list(pf.particles.items()) if v > 0]), places=4) # Slightly more particles lie on the same (row, col) locations

    self.assertAlmostEqual(4, pf.particles[(3,9)], places=4)
    self.assertAlmostEqual(0, pf.particles[(2,10)], places=4) # 0 --> 0
    self.assertAlmostEqual(5, pf.particles[(8,4)], places=4)
    self.assertAlmostEqual(3, pf.particles[(12,6)], places=4)
    self.assertAlmostEqual(0, pf.particles[(7,8)], places=4)
    self.assertAlmostEqual(2, pf.particles[(11,6)], places=4)
    self.assertAlmostEqual(0, pf.particles[(18,7)], places=4) # 0 --> 1
    self.assertAlmostEqual(1, pf.particles[(20,5)], places=4) # 1 --> 0

  @graded()
  def test_2(self):
    """3a-2-basic:  3a basic test for PF observe AND elapseTime"""
    random.seed(3)
    pf = model.ParticleFilter(30, 13)
    self.assertAlmostEqual(69,  len([k for k,v in list(pf.particles.items()) if v > 0]), places=4) # This should not fail unless your code changed the random initialization code.

    pf.elapseTime()
    self.assertAlmostEqual(58, len([k for k,v in list(pf.particles.items()) if v > 0]), places=4) # Most particles lie on the same (row, col) locations
    pf.observe(555, 193, 800)

    self.assertAlmostEqual(200, sum(pf.particles.values()), places=4) # Do not lose particles
    self.assertAlmostEqual(2, len([k for k,v in list(pf.particles.items()) if v > 0]), places=4) # Most particles lie on the same (row, col) locations
    self.assertAlmostEqual(0.025, pf.belief.getProb(20, 4), places=4)
    self.assertAlmostEqual(0.0, pf.belief.getProb(21, 5), places=4)
    self.assertAlmostEqual(0.0, pf.belief.getProb(21, 6), places=4)
    self.assertAlmostEqual(0.975, pf.belief.getProb(22, 6), places=4)
    self.assertAlmostEqual(0.0, pf.belief.getProb(22, 7), places=4)

    pf.elapseTime()
    self.assertAlmostEqual(4, len([k for k,v in list(pf.particles.items()) if v > 0]), places=4) # Most particles lie on the same (row, col) locations

    pf.observe(660, 193, 50)
    self.assertAlmostEqual(0.0, pf.belief.getProb(20, 4), places=4)
    self.assertAlmostEqual(0.0, pf.belief.getProb(21, 5), places=4)
    self.assertAlmostEqual(0.0, pf.belief.getProb(21, 6), places=4)
    self.assertAlmostEqual(0.0, pf.belief.getProb(22, 6), places=4)
    self.assertAlmostEqual(1.0, pf.belief.getProb(22, 7), places=4)

  @graded(is_hidden=True)
  def test_3i(self):
    """3a-3i-hidden:  3a advanced test for PF observe"""
    random.seed(34)
    A = 30
    B = 30
    pf = model.ParticleFilter(A, B)

    N = 50
    p_values = []
    for i in range(N):
      SEED_MODE = 1000 # setup the random seed for fairness
      seed = int(random.random() * SEED_MODE)
      nextSeed = int(random.random() * SEED_MODE)

      a = int(random.random() * 30)
      b = int(random.random() * 5)
      c = int(random.random() * 30)

      random.seed(seed)
      pf.observe(a, b, c)
      random.seed(seed)
      for d in range(A):
        for e in range(B):
          p_values.append(pf.belief.getProb(d, e))
      random.seed(nextSeed)

  @graded(is_hidden=True)
  def test_3ii(self):
    """3a-3ii-hidden:  3a test for pdf ordering"""
    random.seed(34)

    oldpdf = util.pdf
    del util.pdf
    def pdf(a, b, c): # You can't swap a and c now!
      return a + b
    util.pdf = pdf

    A = 30
    B = 30
    random.seed(34)
    pf = model.ParticleFilter(A, B)

    N = 50
    p_values = []
    for i in range(N):
      SEED_MODE = 1000 # setup the random seed for fairness
      seed = int(random.random() * SEED_MODE)
      nextSeed = int(random.random() * SEED_MODE)

      a = int(random.random() * 30)
      b = int(random.random() * 5)
      c = int(random.random() * 30)

      random.seed(seed)
      pf.observe(a, b, c)
      for d in range(A):
        for e in range(B):
          p_values.append(pf.belief.getProb(d, e))
      random.seed(nextSeed)

    util.pdf = oldpdf # fix the pdf

  @graded(is_hidden=True)
  def test_4(self):
    """3a-4-hidden:  advanced test for PF elapseTime"""
    A = 30
    B = 30
    random.seed(35)
    pf = model.ParticleFilter(A, B)

    N1 = 20
    N2 = 400
    p_values = []
    for i in range(N1):
      SEED_MODE = 1000 # setup the random seed for fairness
      seed = int(random.random() * SEED_MODE)
      nextSeed = int(random.random() * SEED_MODE)

      random.seed(seed)
      pf.elapseTime()

      for i in range(N2):
        d = int(random.random() * A)
        e = int(random.random() * B)
        p_values.append(pf.belief.getProb(d, e))
      random.seed(nextSeed)

  @graded(is_hidden=True)
  def test_5(self):
    """3a-5-hidden:  advanced test for PF observe AND elapseTime"""
    A = 30
    B = 30
    random.seed(36)
    pf = model.ParticleFilter(A, B)

    N1 = 20
    N2 = 400
    p_values = []
    for i in range(N1):
      SEED_MODE = 1000 # setup the random seed for fairness
      seed = int(random.random() * SEED_MODE)
      seed2 = int(random.random() * SEED_MODE)
      nextSeed = int(random.random() * SEED_MODE)

      random.seed(seed)
      pf.elapseTime()

      a = int(random.random() * 5 * A)
      b = int(random.random() * 5)
      c = int(random.random() * 5 * A)

      random.seed(seed2)
      pf.observe(a, b, c)
      for i in range(N2):
        d = int(random.random() * A)
        e = int(random.random() * B)
        p_values.append(pf.belief.getProb(d, e))
      random.seed(nextSeed)

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