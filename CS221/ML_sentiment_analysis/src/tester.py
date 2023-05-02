#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect, collections, os, pickle, gzip
from testerUtil import graded, CourseTestRunner, GradedTestCase
from util import *

import model

# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
# TESTS #
class Test_1a(GradedTestCase):
    @graded(timeout=1)
    def test_0(self):
        """1a-0-basic:  Basic test case."""
        ans = {"a": 2, "b": 1}
        self.assertEqual(ans, model.extractWordFeatures("a b a"))

    @graded(timeout=1, is_hidden=True)
    def test_1(self):
        """1a-1-hidden:  Test multiple instances of the same word in a sentence."""
        random.seed(42)
        for i in range(10):
            sentence = " ".join(
                [random.choice(["a", "aa", "ab", "b", "c"]) for _ in range(100)]
            )
            self.compare_with_solution_or_wait(
                model, "extractWordFeatures", lambda f: f(sentence)
            )


class Test_1b(GradedTestCase):
    @graded(timeout=1)
    def test_0(self):
        """1b-0-basic:  Basic sanity check for learning correct weights on two training and testing examples each."""
        trainExamples = (("hello world", 1), ("goodnight moon", -1))
        validationExamples = (("hello", 1), ("moon", -1))
        featureExtractor = model.extractWordFeatures
        weights = model.learnPredictor(
            trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01
        )
        self.assertLess(0, weights["hello"])
        self.assertGreater(0, weights["moon"])

    @graded(timeout=1)
    def test_1(self):
        """1b-1-basic:  Test correct overriding of positive weight due to one negative instance with repeated words."""
        trainExamples = (("hi bye", 1), ("hi hi", -1))
        validationExamples = (("hi", -1), ("bye", 1))
        featureExtractor = model.extractWordFeatures
        weights = model.learnPredictor(
            trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01
        )
        self.assertGreater(0, weights["hi"])
        self.assertLess(0, weights["bye"])

    @graded(timeout=8)
    def test_2(self):
        """1b-2-basic:  Test classifier on real polarity dev dataset."""
        trainExamples = readExamples("polarity.train")
        devExamples = readExamples("polarity.dev")
        featureExtractor = model.extractWordFeatures
        weights = model.learnPredictor(
            trainExamples, devExamples, featureExtractor, numEpochs=20, eta=0.01
        )
        outputWeights(weights, "weights")
        outputErrorAnalysis(
            devExamples, featureExtractor, weights, "error-analysis"
        )  # Use this to debug
        trainError = evaluatePredictor(
            trainExamples,
            lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
        )
        devError = evaluatePredictor(
            devExamples,
            lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
        )
        print(("Official: train error = %s, dev error = %s" % (trainError, devError)))
        self.assertGreater(0.04, trainError)
        self.assertGreater(0.30, devError)


class Test_1c(GradedTestCase):
    @graded(timeout=1)
    def test_0(self):
        """1c-0-basic:  test correct generation of random dataset labels"""
        weights = {"hello": 1, "world": 1}
        data = model.generateDataset(5, weights)
        for datapt in data:
            self.assertEqual((dotProduct(datapt[0], weights) >= 0), (datapt[1] == 1))

    @graded(timeout=1)
    def test_1(self):
        """1c-1-basic:  test that the randomly generated example actually coincides with the given weights"""
        weights = {}
        for i in range(100):
            weights[str(i + 0.1)] = 1
        data = model.generateDataset(100, weights)
        for datapt in data:
            self.assertEqual(False, dotProduct(datapt[0], weights) == 0)


class Test_1d(GradedTestCase):
    @graded(timeout=1)
    def test_0(self):
        """1d-0-basic:  test basic character n-gram features"""
        fe = model.extractCharacterFeatures(3)
        sentence = "hello world"
        ans = {
            "hel": 1,
            "ell": 1,
            "llo": 1,
            "low": 1,
            "owo": 1,
            "wor": 1,
            "orl": 1,
            "rld": 1,
        }
        self.assertEqual(ans, fe(sentence))

    @graded(timeout=1, is_hidden=True)
    def test_1(self):
        """1d-1-hidden:  test feature extraction on repeated character n-grams"""
        random.seed(42)

        for i in range(10):
            sentence = " ".join(
                [random.choice(["a", "aa", "ab", "b", "c"]) for _ in range(100)]
            )
            for n in range(1, 4):
                self.compare_with_solution_or_wait(
                    model, "extractCharacterFeatures", lambda f: f(n)(sentence)
                )


class Test_2b(GradedTestCase):
    @graded(timeout=1)
    def test_0(self):
        """2b-0-basic:  test basic k-means on hardcoded datapoints."""
        random.seed(42)
        x1 = {0: 0, 1: 0}
        x2 = {0: 0, 1: 1}
        x3 = {0: 0, 1: 2}
        x4 = {0: 0, 1: 3}
        x5 = {0: 0, 1: 4}
        x6 = {0: 0, 1: 5}
        examples = [x1, x2, x3, x4, x5, x6]
        centers, projects, totalCost = model.kmeans(examples, 2, maxEpochs=10)
        # (there are two stable centroid locations)
        self.assertEqual(
            True,
            round(totalCost, 3) == 4
            or round(totalCost, 3) == 5.5
            or round(totalCost, 3) == 5.0,
        )

    @graded(timeout=1, is_hidden=True)
    def test_1(self):
        """2b-1-hidden:  test stability of cluster projects."""
        random.seed(42)
        K = 6
        bestCenters = None
        bestProjects = None
        bestTotalCost = None
        examples = generateClusteringExamples(
            numExamples=1000, numWordsPerTopic=3, numFillerWords=1000
        )
        centers, projects, totalCost = model.kmeans(examples, K, maxEpochs=100)

    @graded(timeout=1, is_hidden=True)
    def test_2(self):
        """2b-2-hidden:  test stability of cluster locations."""
        random.seed(42)
        K = 6
        bestCenters = None
        bestProjects = None
        bestTotalCost = None
        examples = generateClusteringExamples(
            numExamples=1000, numWordsPerTopic=3, numFillerWords=1000
        )
        centers, projects, totalCost = model.kmeans(examples, K, maxEpochs=100)

    @graded(timeout=5, is_hidden=True)
    def test_3(self):
        """2b-3-hidden:  scaling kmean."""
        random.seed(42)
        K = 6
        bestCenters = None
        bestProjects = None
        bestTotalCost = None
        examples = generateClusteringExamples(
            numExamples=10000, numWordsPerTopic=3, numFillerWords=10000
        )
        centers, projects, totalCost = model.kmeans(examples, K, maxEpochs=100)
        self.assertEqual(True, totalCost < 10e6)


def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case", nargs="?", default="all")
    test_id = parser.parse_args().test_case

    project = unittest.TestSuite()
    if test_id != "all":
        project.addTest(getTestCaseForTestID(test_id))
    else:
        project.addTests(
            unittest.defaultTestLoader.discover(".", pattern="tester.py")
        )
    CourseTestRunner().run(project)
