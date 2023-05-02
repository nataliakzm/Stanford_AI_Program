#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar, DefaultDict
from util import *
import itertools
import collections
from collections import Counter

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    wordDict=collections.defaultdict(float)
    for word in x.split():
        wordDict[word]+=1
    return wordDict

T = TypeVar("T")

def learnPredictor(
    trainExamples: List[Tuple[T, int]],
    validationExamples: List[Tuple[T, int]],
    featureExtractor: Callable[[T], FeatureVector],
    numEpochs: int,
    eta: float,
) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.
    """
    weights = {}  # feature => weight
    def predict(x):
        phi=featureExtractor(x)
        if dotProduct(weights,phi)<0.0:
            return -1
        else:
            return 1
    for i in range(numEpochs):
        for item in trainExamples:
            x,y=item
            phi=featureExtractor(x)
            temp=dotProduct(weights,phi)*y
            if temp < 1:increment(weights,-eta*-y,phi)
       # print("Iteration:%s, Training error:%s, Test error:%s"%(i,evaluatePredictor(trainExamples,predict),evaluatePredictor(testExamples,predict)))
    return weights

def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    def generateExample() -> Tuple[Dict[str, int], int]:
        phi = None
        y = None
        phi={}
        for item in random.sample(list(weights),random.randint(1,len(weights))):
            phi[item]=random.randint(1,100)
        y=1 if dotProduct(weights,phi)>1 else 0       
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x):
        returnDict=collections.defaultdict(int)
        x=x.replace(' ','')
        for i in range(0,len(x)-(n-1)):
            returnDict[x[i:i+n]]+=1
        return returnDict
    return extract


############################################################
# To run this function, run the command from termial with `n` replaced
# $ python -c "from submission import *; testValuesOfN(n)"

def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    """
    trainExamples = readExamples("polarity.train")
    validationExamples = readExamples("polarity.dev")
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(
        trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01
    )
    outputWeights(weights, "weights")
    outputErrorAnalysis(
        validationExamples, featureExtractor, weights, "error-analysis"
    )  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    validationError = evaluatePredictor(
        validationExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    print(
        (
            "Official: train error = %s, validation error = %s"
            % (trainError, validationError)
        )
    )

def kmeans(
    examples: List[Dict[str, float]], K: int, maxEpochs: int
) -> Tuple[List, List, float]:
    """
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of projects (i.e. if examples[i] belongs to centers[j], then projects[i] = j),
            final reconstruction loss)
    """
    def distance(x1, x2):
        result = 0
        for f, v in x2.items():
            result += (x1.get(f, 0) - v) ** 2
        return result

    def average(points):
        n = float(len(points))
        result = {}
        for p in points:
            increment(result, 1 / n, p)
        return result

    centroids = random.sample(examples, K)
    old_projects = []
    for _ in range(maxEpochs):
        center_points_pair = []
        projects = []
        totalCost = 0
# ## map phase:
        for p in examples:
            dis = [distance(c, p) for c in centroids]
            newCenter = dis.index(min(dis))
            projects.append(newCenter)
            totalCost += min(dis)
#           print dis, newCenter
            center_points_pair.append((newCenter, p))
        if projects == old_projects:
            break
        else:
            old_projects = list(projects)
        center_points_pair = sorted(center_points_pair, key=lambda item: item[0])
# ## reduce phase with groupby() and average():
        new_centroids = []
        for key, kpList in itertools.groupby(center_points_pair, key=lambda item:item[0]):
            pList = [ kp[1] for kp in kpList]
            new_centroids.append(average(pList))

#       print 'new centroids are', new_centroids
        centroids = new_centroids
    return centroids, projects, totalCost