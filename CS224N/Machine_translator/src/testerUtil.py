#!/usr/bin/env python3
from functools import wraps
import sys, signal, os, time, datetime, unittest, importlib.util, argparse, json, traceback

use_solution = importlib.util.find_spec('solution') is not None
if use_solution:
  import solution

# TEST DECORATORS #
class graded():
  def __init__(self,
               leaderboard_col_name = None,
               leaderboard_sort_order='desc',
               is_hidden = False,
               is_extra_credit = False,
               timeout = 5,
               after_published = False,
               hide_errors = False,
               model_feedback=None):
    self.leaderboard_col_name = leaderboard_col_name
    self.leaderboard_sort_order = leaderboard_sort_order
    self.is_hidden = is_hidden
    self.is_extra_credit = is_extra_credit
    self.timeout = timeout
    self.after_published = after_published
    self.hide_errors = hide_errors
    self.model_feedback = model_feedback

  def __call__(self, func):
    func = timeout_func(self.timeout)(func)
    func.__timeout__ = self.timeout
    func.__is_hidden__ = self.is_hidden
    func.__after_published__ = self.after_published
    func.__hide_errors__ = self.hide_errors
    func.__is_extra_credit__ = self.is_extra_credit
    func.__leaderboard_col_name__ = self.leaderboard_col_name
    func.__leaderboard_sort_order__ = self.leaderboard_sort_order
    func.__model_feedback__ = self.model_feedback
    @wraps(func)
    def wrapper(*args, **kwargs):
      
      # Method for storing result of leaderboard after test completes
      def set_leaderboard_value(x):
        args[0].__leaderboard_value__ = x
      
      # Method for storing elapsed time after test completes
      def set_elapsed(elapsed):
        args[0].__elapsed__ = elapsed
      
      if self.leaderboard_col_name:
        kwargs['set_leaderboard_value'] = set_leaderboard_value
      args[0].starttime = time.perf_counter()
      result = func(*args, **kwargs) 
      endtime = time.perf_counter()
      set_elapsed(endtime - args[0].starttime)
      if self.is_hidden and not use_solution:
        # SKip the test if it is hidden and the solution is not present
        args[0].skipTest('Hidden tests are skipped if the solution is not present.')
      return result
    return wrapper

class timeout_func:
  def __init__(self, maxSeconds):
    self.maxSeconds = maxSeconds

  def __call__(self, func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      # Windows signal library does not have a timer interrupt
      if os.name != 'nt':
        # CAUTION: This overrides the previous timer.  Make sure to cleanup after.
        # Define the timeout function.
        def handle_timeout(signum, frame):
          args[0].fail(f'Test case timed out.  Max time: {self.maxSeconds} seconds')
        # Clear prior alarms
        signal.alarm(0)
        # Assign the timeout function
        signal.signal(signal.SIGALRM, handle_timeout)
        # Set the alarm
        signal.alarm(self.maxSeconds)
        result = func(*args, **kwargs)
        # Cleanup for potential future timers
        signal.alarm(0)
      else:
        result = func(*args, **kwargs)
      return result
    return wrapper

def blockPrint():
  # Disable
  sys.stdout = open(os.devnull, 'w')

def enablePrint():
  # Restore
  sys.stdout = sys.__stdout__

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class GradedTestCase(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.isWritten = False
    self.loadWeights()

  def loadWeights(self):
    if os.path.exists('points.json'): path = 'points.json'
    else: path = '../points.json'
    with open(path) as f:
      self.weights = json.load(f)

  def id(self):
    return self.shortDescription().split(':')[0]

  @property
  def weight(self):
    return self.weights[self.id()]['points']

  @property
  def leaderboardColName(self):
    return getattr(getattr(self, self._testMethodName), '__leaderboard_col_name__', None)

  @property
  def leaderboardValue(self):
    return self.__leaderboard_value__ if hasattr(self, '__leaderboard_value__') else None

  @property
  def isHidden(self):
    return getattr(getattr(self, self._testMethodName), '__is_hidden__', None)

  @property
  def hideErrors(self):
    return getattr(getattr(self, self._testMethodName), '__hide_errors__', None)

  @property
  def afterPublished(self):
    return getattr(getattr(self, self._testMethodName), '__after_published__', None)

  @property
  def isExtraCredit(self):
    return getattr(getattr(self, self._testMethodName), '__is_extra_credit__', None)

  @property
  def modelFeedback(self):
    return getattr(getattr(self, self._testMethodName), '__model_feedback__', None)

  @property
  def timeout(self):
    return getattr(getattr(self, self._testMethodName), '__timeout__', None)

  @property
  def elapsed(self):
    return self.__elapsed__ if hasattr(self, '__elapsed__') else time.perf_counter()-self.starttime

  @property
  def earned(self):
    return self.__earned__ if hasattr(self, '__earned__') else None

  @earned.setter
  def earned(self, earned):
    self.__earned__ = earned

  def run_with_solution_if_possible(self, run_model, func):
    if use_solution:
      return func(solution)
    else:
      return func(run_model)

  def compare_with_solution_or_wait(self, run_model, func_name, comp):
    start = time.perf_counter()
    ans2 = comp(getattr(run_model, func_name))
    end = time.perf_counter()
    if use_solution:
      ans1 = comp(getattr(solution, func_name))
      self.assertEqual(ans1, ans2)
    else:
      # If not using the solution, double the time to make the runtime more realistic
      time.sleep(end-start)

class GradescopeTestResult(unittest.TestResult):
  def __init__(self, stream):
    super().__init__(stream)
    self.stream = stream

  def startTestRun(self):
    super().startTestRun()
    self.results = {
      'tests':[],
      'leaderboard':[]
    }

  def stopTestRun(self):
    super().stopTestRun()
    self.stream.write(json.dumps(self.results)+'\n')

  def addSuccess(self, test):
    super().addSuccess(test)
    test.earned = test.weight if not test.earned else test.earned
    self.storeResult(test, True)

  def addFailure(self, test, err):
    super().addFailure(test, err)
    self.storeResult(test, False, err=err)

  def addError(self, test, err):
    super().addError(test, err)
    self.storeResult(test, False, err=err)

  def addSkip(self, test, reason):
    super().addSkip(test, reason)

  def storeResult(self, test, isSuccess, err=None):
    earned = test.earned if isSuccess else 0
    visibility = 'after_published' if test.afterPublished else 'visible'
    test_result = {
      'score':earned,
      'max_score':test.weight,
      'number':test.id(),
      'name':test.shortDescription().split(':')[1].strip(),
      'visibility':visibility,
      'extra_data':{'is_extra_credit':test.isExtraCredit}
    }
    test_result['output'] = ''
    if test.modelFeedback is not None:
      test_result['output'] += test.modelFeedback + '\n'
    if err is not None and not test.hideErrors:
      test_result['output'] += str(err[0]) + ':  '
      test_result['output'] += str(err[1]) + '\n'
      test_result['output'] += ''.join(traceback.format_tb(err[2]))

    self.results['tests'].append(test_result)
    if test.leaderboardValue is not None:
      self.results['leaderboard'].append({'name':test.leaderboardColName,'value':test.leaderboardValue})

class ModelTestResult(unittest.TestResult):
  """ 
  Attributes:
    stream: io.TextIOBase. This is a simple text stream, which could be a file
      or in-memory stream.  The results of the tests are written to this stream
      in a human-readable format.
    earned_points: int.  The total number of points earned for these tests
    max_points: int.  The total number of points available for these tests.
    earned_extra_credit: int.  The total number of points earned for these tests
    max_extra_credit: int.  The total number of points available for these tests.
  """
  def __init__(self, stream):
    super().__init__(stream)
    self.stream = stream

    self.earned_points = 0
    self.max_points = 0

    self.earned_extra_credit = 0
    self.max_extra_credit = 0

  def startTestRun(self):
    self.stream.write('========== START GRADING\n')

  def stopTestRun(self):
    if not use_solution:
      self.stream.write('Note that the hidden test cases do not check for correctness.\n')
      self.stream.write('They are provided for you to verify that the functions do not crash and run within the time limit.\n')
      self.stream.write('Points for these parts not assigned by the grader unless the solution is present (indicated by "???").\n')
    self.stream.write(f'========== END GRADING [{self.earned_points}/{self.max_points} points + {self.earned_extra_credit}/{self.max_extra_credit} extra credit]\n')

  def startTest(self, test):
    super().startTest(test)
    weight = 0 if test.isHidden and not use_solution else test.weight
    if test.isExtraCredit:
      self.max_extra_credit += weight
    else:
      self.max_points += weight
    self.stream.write('----- START '+test.shortDescription()+'\n')

  def addSuccess(self, test):
    super().addSuccess(test)
    test.earned = test.weight if not test.earned else test.earned
    self.writeTestResults(test, True)

  def addFailure(self, test, err):
    super().addFailure(test, err)
    print(err[0])
    print(err[1])
    traceback.print_tb(err[2])
    self.writeTestResults(test, False)

  def addError(self, test, err):
    super().addError(test, err)
    print(err[0])
    print(err[1])
    traceback.print_tb(err[2])
    self.writeTestResults(test, False)

  def addSkip(self, test, reason):
    self.writeTestResults(test, False)

  def writeTestResults(self, test, isSuccess):
    earned = '???' if test.isHidden and not use_solution else (test.earned if isSuccess else 0)
    self.earned_points += test.earned if isSuccess and not test.isExtraCredit else 0
    self.earned_extra_credit += test.earned if isSuccess and test.isExtraCredit else 0
    hidden_blurb = ' (hidden test ungraded)' if test.isHidden and not use_solution else ''
    self.stream.write(f'----- END {test.id()} [took {datetime.timedelta(seconds=test.elapsed)} (max allowed {test.timeout} seconds), {earned}/{test.weight} points]'+hidden_blurb+'\n\n')

class CourseTestRunner():
  """"""

  def __init__(self, stream=None, gradescope=False, create_latex=False):
    """"""
    if stream is None:
      stream = sys.stdout
    self.stream = stream
    if gradescope:
      self.resultclass = GradescopeTestResult
    else:
      self.resultclass = ModelTestResult
    self.gradescope = gradescope

  def run(self, test):
    result = self.resultclass(self.stream)
    if self.gradescope:
      with HiddenPrints():
        result.startTestRun()
        test(result)
        result.stopTestRun()
    else:
      result.startTestRun()
      test(result)
      result.stopTestRun()

    return result

if __name__ == '__main__':
  assignment = unittest.TestSuite()
  assignment.addTest(unittest.defaultTestLoader.discover('.', pattern='tester.py'))
  CourseTestRunner(gradescope=True).run(assignment)