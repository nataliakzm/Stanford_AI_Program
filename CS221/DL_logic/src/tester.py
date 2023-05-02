#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect, collections, os, pickle, gzip
from testerUtil import graded, CourseTestRunner, GradedTestCase

from logic import *
import nlparser
import model

# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
class TestCase_A7(GradedTestCase):
  def checkFormula(self, name, predForm, preconditionForm=None):
    filename = os.path.join('models', name + '.pklz')
    objects, targetModels = pickle.load(gzip.open(filename))
    # If preconditionion exists, change the formula to
    preconditionPredForm = And(preconditionForm, predForm) if preconditionForm else predForm
    predModels = performModelChecking([preconditionPredForm], findAll=True, objects=objects)
    ok = True
    def hashkey(model): return tuple(sorted(str(atom) for atom in model))
    targetModelSet = set(hashkey(model) for model in targetModels)
    predModelSet = set(hashkey(model) for model in predModels)
    for model in targetModels:
      self.assertTrue(hashkey(model) in predModelSet, msg=f'Your formula ({predForm}) says the following model is FALSE, but it should be TRUE:\n{printModel(model, ret=True)}')
    for model in predModels:
      self.assertTrue(hashkey(model) in targetModelSet, msg=f'Your formula ({predForm}) says the following model is TRUE, but it should be FALSE:\n{printModel(model, ret=True)}')
    print(f'You matched the {len(targetModels)} models')
    print(f'Example model: {rstr(random.choice(targetModels))}')

  def check(self, part, name, numForms, predictionFunc):
    predForms, predQuery = predictionFunc()
    if len(predForms) < numForms:
      tester.fail("Wanted %d formulas, but got %d formulas:" % (numForms, len(predForms)))
      for form in predForms: print(('-', form))
      return
    if part == 'all':
      self.checkFormula(name + '-all', AndList(predForms))
    elif part == 'run':
      kb = createModelCheckingKB()

      # Need to tell the KB about the objects to do model checking
      filename = os.path.join('models', name + '-all.pklz')
      objects, targetModels = pickle.load(gzip.open(filename))
      for obj in objects:
        kb.tell(Atom('Object', obj))

      # Add the formulas
      for predForm in predForms:
        response = kb.tell(predForm)
        showKBResponse(response)
        self.assertEqual(CONTINGENT, response.status)
      response = kb.ask(predQuery)
      showKBResponse(response)

    else:  # Check the part-th formula
      self.checkFormula(name + '-' + str(part), predForms[part])

  def getTopDerivation(self, sentence, languageProcessor, grammar):
    print()
    print(('>>>', sentence))
    utterance = nlparser.Utterance(sentence, languageProcessor)
    print(('Utterance:', utterance))
    derivations = nlparser.parseUtterance(utterance, grammar, verbose=0)
    if not derivations:
      raise Exception('Error: Parsing failed. (0 derivations)')
    return derivations[0].form

  def compareExtraCreditKnowledgeBase(self, examples, ruleCreator):
    # Test the logical forms by querying the knowledge base.
    kb = createModelCheckingKB()
    #kb = createResolutionKB()
    languageProcessor = nlparser.createBaseLanguageProcessor()
    # Need to tell kb about objects
    for obj in nlparser.BASE_OBJECTS:
      kb.tell(Atom('Object', obj.lower()))

    # Parse!
    grammar = nlparser.createBaseEnglishGrammar() + [ruleCreator()]
    for sentence, expectedResult in examples:
      mode, formula = self.getTopDerivation(sentence, languageProcessor, grammar)
      print(('The parser returns:', (mode, formula)))
      self.assertEqual(expectedResult[0], mode)
      if mode == 'tell':  response = kb.tell(formula)
      if mode == 'ask':   response = kb.ask(formula)
      print(('Knowledge base returns:', response))
      self.assertEqual(expectedResult[1], response.status)


# TESTS #
class Test_1a(TestCase_A7):
  @graded()
  def test_0(self):
    """1a-0-basic:  Test formula 1a implementation"""
    self.checkFormula('1a', model.formula1a())
class Test_1b(TestCase_A7):
  @graded()
  def test_0(self):
    """1b-0-basic:  Test formula 1b implementation"""
    self.checkFormula('1b', model.formula1b())
class Test_1c(TestCase_A7):
  @graded()
  def test_0(self):
    """1c-0-basic:  Test formula 1c implementation"""
    self.checkFormula('1c', model.formula1c())
class Test_2a(TestCase_A7):
  @graded()
  def test_0(self):
    """2a-0-basic:  Test formula 2a implementation"""
    formula2a_precondition = AntiReflexive('Mother')
    self.checkFormula('2a', model.formula2a(), formula2a_precondition)
class Test_2b(TestCase_A7):
  @graded()
  def test_0(self):
    """2b-0-basic:  Test formula 2b implementation"""
    formula2b_precondition = AntiReflexive('Child')
    self.checkFormula('2b', model.formula2b(), formula2b_precondition)
class Test_2c(TestCase_A7):
  @graded()
  def test_0(self):
    """2c-0-basic:  Test formula 2c implementation"""
    formula2c_precondition = AntiReflexive('Child')
    self.checkFormula('2c', model.formula2c(), formula2c_precondition)
class Test_2d(TestCase_A7):
  @graded()
  def test_0(self):
    """2d-0-basic:  Test formula 2d implementation"""
    formula2d_precondition = AntiReflexive('Parent')
    self.checkFormula('2d', model.formula2d(), formula2d_precondition)
class Test_3a(TestCase_A7):
  @graded(timeout=10000)
  def test_0(self):
    """3a-0-basic:  test implementation of statement 0 for 3a"""
    self.check(0, '3a', 6, model.liar)

  @graded(timeout=10000)
  def test_1(self):
    """3a-1-basic:  test implementation of statement 1 for 3a"""
    self.check(1, '3a', 6, model.liar)

  @graded(timeout=10000)
  def test_2(self):
    """3a-2-basic:  test implementation of statement 2 for 3a"""
    self.check(2, '3a', 6, model.liar)

  @graded(timeout=10000)
  def test_3(self):
    """3a-3-basic:  test implementation of statement 3 for 3a"""
    self.check(3, '3a', 6, model.liar)

  @graded(timeout=10000)
  def test_4(self):
    """3a-4-basic:  test implementation of statement 4 for 3a"""
    self.check(4, '3a', 6, model.liar)

  @graded(timeout=10000)
  def test_5(self):
    """3a-5-basic:  test implementation of statement 5 for 3a"""
    self.check(5, '3a', 6, model.liar)

  @graded(timeout=10000)
  def test_all(self):
    """3a-all-basic:  test implementation of all for 3a"""
    self.check('all', '3a', 6, model.liar)

  @graded(timeout=10000)
  def test_run(self):
    """3a-run-basic:  test implementation of run for 3a"""
    self.check('run', '3a', 6, model.liar)
class Test_4a(TestCase_A7):
  @graded(timeout=10000)
  def test_0(self):
    """4a-0-basic:  test implementation of statement 0 for 4a"""
    self.check(0, '4a', 6, model.ints)

  @graded(timeout=10000)
  def test_1(self):
    """4a-1-basic:  test implementation of statement 1 for 4a"""
    self.check(1, '4a', 6, model.ints)

  @graded(timeout=10000)
  def test_2(self):
    """4a-2-basic:  test implementation of statement 2 for 4a"""
    self.check(2, '4a', 6, model.ints)

  @graded(timeout=10000)
  def test_3(self):
    """4a-3-basic:  test implementation of statement 3 for 4a"""
    self.check(3, '4a', 6, model.ints)

  @graded(timeout=10000)
  def test_4(self):
    """4a-4-basic:  test implementation of statement 4 for 4a"""
    self.check(4, '4a', 6, model.ints)

  @graded(timeout=10000)
  def test_5(self):
    """4a-5-basic:  test implementation of statement 5 for 4a"""
    self.check(5, '4a', 6, model.ints)

  @graded(timeout=10000)
  def test_all(self):
    """4a-all-basic:  test implementation of all for 4a"""
    self.check('all', '4a', 6, model.ints)

  @graded(timeout=10000)
  def test_run(self):
    """4a-run-basic:  test implementation of run for 4a"""
    self.check('run', '4a', 6, model.ints)
class Test_5a(TestCase_A7):
  @graded(timeout=60, is_extra_credit=True)
  def test_0(self):
    """5a-0-basic:  Check basic behavior of rule"""
    examples = [
        ('Every person likes some cat.', ('tell', CONTINGENT)),
        ('Every cat is a mammal.', ('tell', CONTINGENT)),
        ('Every person likes some mammal?', ('ask', ENTAILMENT)),
    ]
    self.compareExtraCreditKnowledgeBase(examples, model.createRule1)

  @graded(timeout=60, is_extra_credit=True)
  def test_1(self):
    """5a-1-basic:  Check basic behavior of rule"""
    examples = [
        ('Every person likes some cat.', ('tell', CONTINGENT)),
        ('Every tabby is a cat.', ('tell', CONTINGENT)),
        ('Every person likes some tabby?', ('ask', CONTINGENT)),
    ]
    self.compareExtraCreditKnowledgeBase(examples, model.createRule1)

  @graded(timeout=60, is_extra_credit=True)
  def test_2(self):
    """5a-2-basic:  Check basic behavior of rule"""
    examples = [
        ('Every person likes some cat.', ('tell', CONTINGENT)),
        ('Every person is a mammal.', ('tell', CONTINGENT)),
        ('Every mammal likes some cat?', ('ask', CONTINGENT)),
    ]
    self.compareExtraCreditKnowledgeBase(examples, model.createRule1)

  @graded(timeout=60, is_extra_credit=True)
  def test_3(self):
    """5a-3-basic:  Check basic behavior of rule"""
    examples = [
        ('Every person likes some cat.', ('tell', CONTINGENT)),
        ('Garfield is a cat.', ('tell', CONTINGENT)),
        ('Jon is a person.', ('tell', CONTINGENT)),
        ('Jon likes Garfield?', ('ask', CONTINGENT)),
    ]
    self.compareExtraCreditKnowledgeBase(examples, model.createRule1)

  @graded(timeout=60, is_extra_credit=True)
  def test_4(self):
    """5a-4-basic:  Check basic behavior of rule"""
    examples = [
        ('Every person likes some cat.', ('tell', CONTINGENT)),
        ('Garfield is a cat.', ('tell', CONTINGENT)),
        ('Jon is a person.', ('tell', CONTINGENT)),
        ('Jon likes Garfield?', ('ask', CONTINGENT)),
    ]
    self.compareExtraCreditKnowledgeBase(examples, model.createRule1)
class Test_5b(TestCase_A7):
  @graded(timeout=60, is_extra_credit=True)
  def test_0(self):
    """5b-0-basic:  Check basic behavior of rule"""
    examples = [
        ('There is some cat that every person likes.', ('tell', CONTINGENT)),
        ('Every cat is a mammal.', ('tell', CONTINGENT)),
        ('There is some mammal that every person likes?', ('ask', ENTAILMENT)),
    ]
    self.compareExtraCreditKnowledgeBase(examples, model.createRule2)

  @graded(timeout=60, is_extra_credit=True)
  def test_1(self):
    """5b-1-basic:  Check basic behavior of rule"""
    examples = [
        ('There is some cat that every person likes.', ('tell', CONTINGENT)),
        ('Jon is a person.', ('tell', CONTINGENT)),
        ('Jon likes some cat?', ('ask', ENTAILMENT)),
    ]
    self.compareExtraCreditKnowledgeBase(examples, model.createRule2)

  @graded(timeout=60, is_extra_credit=True)
  def test_2(self):
    """5b-2-basic:  Check basic behavior of rule"""
    examples = [
        ('There is some cat that every person likes.', ('tell', CONTINGENT)),
        ('Garfield is a cat.', ('tell', CONTINGENT)),
        ('Jon is a person.', ('tell', CONTINGENT)),
        ('Jon likes Garfield?', ('ask', CONTINGENT)),
    ]
    self.compareExtraCreditKnowledgeBase(examples, model.createRule2)
class Test_5c(TestCase_A7):
  @graded(timeout=60, is_extra_credit = True)
  def test_0(self):
    """5c-0-basic:  Check basic behavior of rule"""
    examples = [
        ('If a person likes a cat then the former feeds the latter.', ('tell', CONTINGENT)),
        ('Jon is a person.', ('tell', CONTINGENT)),
        ('Jon likes Garfield.', ('tell', CONTINGENT)),
        ('Jon feeds Garfield?', ('ask', CONTINGENT)),
    ]
    self.compareExtraCreditKnowledgeBase(examples, model.createRule3)

  @graded(timeout=60, is_extra_credit = True)
  def test_1(self):
    """5c-1-basic:  Check basic behavior of rule"""
    examples = [
        ('If a person likes a cat then the former feeds the latter.', ('tell', CONTINGENT)),
        ('Jon is a person.', ('tell', CONTINGENT)),
        ('Jon likes Garfield.', ('tell', CONTINGENT)),
        ('Garfield is a cat.', ('tell', CONTINGENT)),
        ('Jon feeds Garfield?', ('ask', ENTAILMENT)),
    ]
    self.compareExtraCreditKnowledgeBase(examples, model.createRule3)

  @graded(timeout=60, is_extra_credit = True)
  def test_2(self):
    """5c-2-basic:  Check basic behavior of rule"""
    examples = [
        ('If a person likes a cat then the former feeds the latter.', ('tell', CONTINGENT)),
        ('Jon likes Garfield.', ('tell', CONTINGENT)),
        ('Garfield is a cat.', ('tell', CONTINGENT)),
        ('Jon feeds Garfield?', ('ask', CONTINGENT)),
    ]
    self.compareExtraCreditKnowledgeBase(examples, model.createRule3)

  @graded(timeout=60, is_extra_credit = True)
  def test_3(self):
    """5c-3-basic:  Check basic behavior of rule"""
    examples = [
        ('If a person likes a cat then the former feeds the latter.', ('tell', CONTINGENT)),
        ('Jon is a person.', ('tell', CONTINGENT)),
        ('Jon likes some cat.', ('tell', CONTINGENT)),
        ('Garfield is a cat.', ('tell', CONTINGENT)),
        ('Jon feeds Garfield?', ('ask', CONTINGENT)),
    ]
    self.compareExtraCreditKnowledgeBase(examples, model.createRule3)

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