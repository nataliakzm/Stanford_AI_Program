#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
from testerUtil import graded, CourseTestRunner, GradedTestCase
import util, wordsegUtil

import model

# HELPER FUNCTIONS FOR CREATING TEST INPUTS #
QUERIES_SEG = [
    "ThestaffofficerandPrinceAndrewmountedtheirhorsesandrodeon",
    "hellothere officerandshort erprince",
    "howdythere",
    "The staff officer and Prince Andrew mounted their horses and rode on.",
    "whatsup",
    "duduandtheprince",
    "duduandtheking",
    "withoutthecourtjester",
    "lightbulbneedschange",
    "imagineallthepeople",
    "thisisnotmybeautifulhouse",
]

QUERIES_INS = [
    "strng",
    "pls",
    "hll thr",
    "whats up",
    "dudu and the prince",
    "frog and the king",
    "ran with the queen and swam with jack",
    "light bulbs need change",
    "ffcr nd prnc ndrw",
    "ffcr nd shrt prnc",
    "ntrntnl",
    "smthng",
    "btfl",
]

QUERIES_BOTH = [
    "stff",
    "hllthr",
    "thffcrndprncndrw",
    "ThstffffcrndPrncndrwmntdthrhrssndrdn",
    "whatsup",
    "ipovercarrierpigeon",
    "aeronauticalengineering",
    "themanwiththegoldeneyeball",
    "lightbulbsneedchange",
    "internationalplease",
    "comevisitnaples",
    "somethingintheway",
    "itselementarymydearwatson",
    "itselementarymyqueen",
    "themanandthewoman",
    "nghlrdy",
    "jointmodelingworks",
    "jointmodelingworkssometimes",
    "jointmodelingsometimesworks",
    "rtfclntllgnc",
]

CORPUS = "leo-will.txt"

_realUnigramCost, _realBigramCost, _possibleFills = None, None, None


def getRealCosts():
    global _realUnigramCost, _realBigramCost, _possibleFills

    if _realUnigramCost is None:
        print(f"Training language cost functions [corpus: {CORPUS}]... ", end="")

        _realUnigramCost, _realBigramCost = wordsegUtil.makeLanguageModels(CORPUS)
        _possibleFills = wordsegUtil.makeInverseRemovalDictionary(CORPUS, "aeiou")

        print("Done!")
        print("")

    return _realUnigramCost, _realBigramCost, _possibleFills


def bigramCost(a, b):
    corpus = [wordsegUtil.SENTENCE_BEGIN] + "beam me up scotty".split()
    if (a, b) in list(zip(corpus, corpus[1:])):
        return 1.0
    else:
        return 1000.0


def possibleFills(x):
    fills = {
        "bm": set(["beam", "bam", "boom"]),
        "m": set(["me", "ma"]),
        "p": set(["up", "oop", "pa", "epe"]),
        "sctty": set(["scotty"]),
    }
    return fills.get(x, set())



# TESTS #
class A2_TestCase(GradedTestCase):
    def setUp(self):
        super().setUp()
        self.unigramCost, self.bigramCost, self.possibleFills = getRealCosts()


class Test_1b(A2_TestCase):
    @graded(timeout=2)
    def test_0(self):
        """1b-0-basic:  simple test case using hand-picked unigram costs."""

        def unigramCost(x):
            if x in ["and", "two", "three", "word", "words"]:
                return 1.0
            else:
                return 1000.0

        self.assertEqual("", model.segmentWords("", unigramCost))
        self.assertEqual("word", model.segmentWords("word", unigramCost))
        self.assertEqual("two words", model.segmentWords("twowords", unigramCost))
        self.assertEqual(
            "and three words", model.segmentWords("andthreewords", unigramCost)
        )

    @graded(timeout=2)
    def test_1(self):
        """1b-1-basic:  simple test case using unigram cost from the corpus"""
        self.assertEqual("word", model.segmentWords("word", self.unigramCost))
        self.assertEqual(
            "two words", model.segmentWords("twowords", self.unigramCost)
        )
        self.assertEqual(
            "and three words",
            model.segmentWords("andthreewords", self.unigramCost),
        )

    @graded(timeout=3, is_hidden=True)
    def test_2(self):
        """1b-2-hidden:"""
        # Word seen in corpus
        solution1 = model.segmentWords("pizza", self.unigramCost)

        # Even long unseen words are preferred to their arbitrary segmentations
        solution2 = model.segmentWords("qqqqq", self.unigramCost)
        solution3 = model.segmentWords("z" * 100, self.unigramCost)

        # But 'a' is a word
        solution4 = model.segmentWords("aa", self.unigramCost)

        # With an apparent crossing point at length 6->7
        solution5 = model.segmentWords("aaaaaa", self.unigramCost)
        solution6 = model.segmentWords("aaaaaaa", self.unigramCost)

    @graded(timeout=3, is_hidden=True)
    def test_3(self):
        """1b-3-hidden:  hidden test case for all queries in QUERIES_SEG"""
        for query in QUERIES_SEG:
            query = wordsegUtil.cleanLine(query)
            parts = wordsegUtil.words(query)
            self.compare_with_solution_or_wait(
                model,
                "segmentWords",
                lambda f: [f(part, self.unigramCost) for part in parts],
            )


class Test_2b(A2_TestCase):
    @graded(timeout=2)
    def test_0(self):
        """2b-0-basic:  simple test case"""

        def bigramCost(a, b):
            corpus = [wordsegUtil.SENTENCE_BEGIN] + "beam me up scotty".split()
            if (a, b) in list(zip(corpus, corpus[1:])):
                return 1.0
            else:
                return 1000.0

        def possibleFills(x):
            fills = {
                "bm": set(["beam", "bam", "boom"]),
                "m": set(["me", "ma"]),
                "p": set(["up", "oop", "pa", "epe"]),
                "sctty": set(["scotty"]),
            }
            return fills.get(x, set())

        self.assertEqual("", model.insertVowels([], bigramCost, possibleFills))
        self.assertEqual(  # No fills
            "zz$z$zz", model.insertVowels(["zz$z$zz"], bigramCost, possibleFills)
        )
        self.assertEqual(
            "beam", model.insertVowels(["bm"], bigramCost, possibleFills)
        )
        self.assertEqual(
            "me up", model.insertVowels(["m", "p"], bigramCost, possibleFills)
        )
        self.assertEqual(
            "beam me up scotty",
            model.insertVowels("bm m p sctty".split(), bigramCost, possibleFills),
        )

    @graded(timeout=2, is_hidden=True)
    def test_1(self):
        """2b-1-hidden:  Simple hidden test case"""
        solution1 = model.insertVowels([], self.bigramCost, self.possibleFills)
        # No fills
        solution2 = model.insertVowels(
            ["zz$z$zz"], self.bigramCost, self.possibleFills
        )
        solution3 = model.insertVowels([""], self.bigramCost, self.possibleFills)
        solution4 = model.insertVowels(
            "wld lk t hv mr lttrs".split(), self.bigramCost, self.possibleFills
        )
        solution5 = model.insertVowels(
            "ngh lrdy".split(), self.bigramCost, self.possibleFills
        )

    @graded(timeout=3, is_hidden=True)
    def test_2(self):
        """2b-2-hidden:  Simple hidden test case."""
        SB = wordsegUtil.SENTENCE_BEGIN

        # Check for correct use of SENTENCE_BEGIN
        def bigramCost(a, b):
            if (a, b) == (SB, "cat"):
                return 5.0
            elif a != SB and b == "dog":
                return 1.0
            else:
                return 1000.0

        solution1 = model.insertVowels(
            ["x"], bigramCost, lambda x: set(["cat", "dog"])
        )

        # Check for non-greediness
        def bigramCost(a, b):
            # Dog over log -- a test poem by rf
            costs = {
                (SB, "cat"): 1.0,  # Always start with cat
                ("cat", "log"): 1.0,  # Locally prefer log
                ("cat", "dog"): 2.0,  # rather than dog
                ("log", "mouse"): 3.0,  # But dog would have been
                ("dog", "mouse"): 1.0,  # better in retrospect
            }
            return costs.get((a, b), 1000.0)

        def fills(x):
            return {
                "x1": set(["cat", "dog"]),
                "x2": set(["log", "dog", "frog"]),
                "x3": set(["mouse", "house", "cat"]),
            }[x]

        solution2 = model.insertVowels("x1 x2 x3".split(), bigramCost, fills)

        # Check for non-trivial long-range dependencies
        def bigramCost(a, b):
            # Dogs over logs -- another test poem by rf
            costs = {
                (SB, "cat"): 1.0,  # Always start with cat
                ("cat", "log1"): 1.0,  # Locally prefer log
                ("cat", "dog1"): 2.0,  # Rather than dog
                ("log20", "mouse"): 1.0,  # And this might even
                ("dog20", "mouse"): 1.0,  # seem to be okay
            }
            for i in range(1, 20):  # But along the way
                #                               Dog's cost will decay
                costs[("log" + str(i), "log" + str(i + 1))] = 0.25
                costs[("dog" + str(i), "dog" + str(i + 1))] = 1.0 / float(i)
            #                               Hooray
            return costs.get((a, b), 1000.0)

        def fills(x):
            f = {
                "x0": set(["cat", "dog"]),
                "x21": set(["mouse", "house", "cat"]),
            }
            for i in range(1, 21):
                f["x" + str(i)] = set(["log" + str(i), "dog" + str(i), "frog"])
            return f[x]

        solution3 = model.insertVowels(
            ["x" + str(i) for i in range(0, 22)], bigramCost, fills
        )

    @graded(timeout=3, is_hidden=True)
    def test_3(self):
        """2b-3-hidden:  hidden test case for all queries in QUERIES_INS"""
        for query in QUERIES_INS:
            query = wordsegUtil.cleanLine(query)
            ws = [wordsegUtil.removeAll(w, "aeiou") for w in wordsegUtil.words(query)]
            self.compare_with_solution_or_wait(
                model,
                "insertVowels",
                lambda f: f(copy.deepcopy(ws), self.bigramCost, self.possibleFills),
            )


class Test_3b(A2_TestCase):
    @graded(timeout=2)
    def test_0(self):
        """3b-0-basic:  Simple test case with hand-picked bigram costs and possible fills."""

        def bigramCost(a, b):
            if b in ["and", "two", "three", "word", "words"]:
                return 1.0
            else:
                return 1000.0

        fills_ = {
            "nd": set(["and"]),
            "tw": set(["two"]),
            "thr": set(["three"]),
            "wrd": set(["word"]),
            "wrds": set(["words"]),
        }
        fills = lambda x: fills_.get(x, set())

        self.assertEqual("", model.segmentAndInsert("", bigramCost, fills))
        self.assertEqual("word", model.segmentAndInsert("wrd", bigramCost, fills))
        self.assertEqual(
            "two words", model.segmentAndInsert("twwrds", bigramCost, fills)
        )
        self.assertEqual(
            "and three words",
            model.segmentAndInsert("ndthrwrds", bigramCost, fills),
        )

    @graded(timeout=2)
    def test_1(self):
        """3b-1-basic:  simple test case with unigram costs as bigram costs"""
        bigramCost = lambda a, b: self.unigramCost(b)

        fills_ = {
            "nd": set(["and"]),
            "tw": set(["two"]),
            "thr": set(["three"]),
            "wrd": set(["word"]),
            "wrds": set(["words"]),
        }
        fills = lambda x: fills_.get(x, set())

        self.assertEqual("word", model.segmentAndInsert("wrd", bigramCost, fills))
        self.assertEqual(
            "two words", model.segmentAndInsert("twwrds", bigramCost, fills)
        )
        self.assertEqual(
            "and three words",
            model.segmentAndInsert("ndthrwrds", bigramCost, fills),
        )

    @graded(timeout=3, is_hidden=True)
    def test_2(self):
        """3b-2-hidden:  hidden test case with unigram costs as bigram costs and additional possible fills."""
        bigramCost = lambda a, b: self.unigramCost(b)
        fills_ = {
            "nd": set(["and"]),
            "tw": set(["two"]),
            "thr": set(["three"]),
            "wrd": set(["word"]),
            "wrds": set(["words"]),
            # Hah!  Hit them with two better words
            "th": set(["the"]),
            "rwrds": set(["rewards"]),
        }
        fills = lambda x: fills_.get(x, set())

        solution1 = model.segmentAndInsert("wrd", bigramCost, fills)
        solution2 = model.segmentAndInsert("twwrds", bigramCost, fills)
        # Waddaya know
        solution3 = model.segmentAndInsert("ndthrwrds", bigramCost, fills)

    @graded(timeout=3, is_hidden=True)
    def test_3(self):
        """3b-3-hidden:  hidden test case with hand-picked bigram costs and possible fills"""

        def bigramCost(a, b):
            corpus = [wordsegUtil.SENTENCE_BEGIN] + "beam me up scotty".split()
            if (a, b) in list(zip(corpus, corpus[1:])):
                return 1.0
            else:
                return 1000.0

        def possibleFills(x):
            fills = {
                "bm": set(["beam", "bam", "boom"]),
                "m": set(["me", "ma"]),
                "p": set(["up", "oop", "pa", "epe"]),
                "sctty": set(["scotty"]),
                "z": set(["ze"]),
            }
            return fills.get(x, set())

        # Ensure no non-word makes it through
        solution1 = model.segmentAndInsert("zzzzz", bigramCost, possibleFills)
        solution2 = model.segmentAndInsert("bm", bigramCost, possibleFills)
        solution3 = model.segmentAndInsert("mp", bigramCost, possibleFills)
        solution4 = model.segmentAndInsert("bmmpsctty", bigramCost, possibleFills)

    @graded(timeout=3, is_hidden=True)
    def test_4(self):
        """3b-4-hidden:  hidden test case for all queries in QUERIES_BOTH with bigram costs and possible fills from the corpus"""
        smoothCost = wordsegUtil.smoothUnigramAndBigram(
            self.unigramCost, self.bigramCost, 0.2
        )
        for query in QUERIES_BOTH:
            query = wordsegUtil.cleanLine(query)
            parts = [
                wordsegUtil.removeAll(w, "aeiou") for w in wordsegUtil.words(query)
            ]
            self.compare_with_solution_or_wait(
                model,
                "segmentAndInsert",
                lambda f: [f(part, smoothCost, self.possibleFills) for part in parts],
            )


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
