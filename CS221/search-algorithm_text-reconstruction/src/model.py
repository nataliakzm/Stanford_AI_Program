from typing import Callable, List, Set

import shell
import util
import wordsegUtil

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query: str, unigramCost: Callable[[str], float]):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        return self.query

    def isEnd(self, state) -> bool:
        return len(state)==0

    def succAndCost(self, state):
        result=[]
        if not self.isEnd(state):
            for i in range(len(state),0,-1):
                action=state[:i]
                cost=self.unigramCost(action)
                remainingText=state[len(action):]
                result.append((action,remainingText,cost))
        return result

def segmentWords(query: str, unigramCost: Callable[[str], float]) -> str:
    if len(query) == 0:
        return ""

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    words=' '.join(ucs.actions)
    return words

class VowelInsertionProblem(util.SearchProblem):
    def __init__(
        self,
        queryWords: List[str],
        bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]],
    ):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        return (self.queryWords[0],0)

    def isEnd(self, state) -> bool:
        return state[1]==len(self.queryWords)-1

    def succAndCost(self, state):
        result = []
        index=state[1]+1
        # temp=self.queryWords[index]
        choices = self.possibleFills(self.queryWords[index]).copy()
        if len(choices)==0:
            choices.add(self.queryWords[index])
        for action in choices:
            cost=self.bigramCost(state[0],action)
            result.append((action, (action,index), cost))
        return result

def insertVowels(
    queryWords: List[str],
    bigramCost: Callable[[str, str], float],
    possibleFills: Callable[[str], Set[str]],
) -> str:
    if len(queryWords)==0:
        return ''
    else:
        queryWords.insert(0,wordsegUtil.SENTENCE_BEGIN)
    ucs=util.UniformCostSearch(verbose=1)
    ucs.solve(VowelInsertionProblem(queryWords,bigramCost,possibleFills))
    words = ' '.join(ucs.actions)
    return words

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(
        self,
        query: str,
        bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]],
    ):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        return (self.query,wordsegUtil.SENTENCE_BEGIN)

    def isEnd(self, state) -> bool:
        if len(state[0])==0:
            return True
        return False

    def succAndCost(self, state):
        result=[]
        for i in range(1,len(state[0])+1):
            subword=state[0][:i]
            remainWord=state[0][i:]
            choices=self.possibleFills(subword).copy()
            for item in choices:
                cost=self.bigramCost(state[1],item)
                result.append((item, (remainWord,item), cost))
        return result

def segmentAndInsert(
    query: str,
    bigramCost: Callable[[str, str], float],
    possibleFills: Callable[[str], Set[str]],
) -> str:
    if len(query) == 0:
        return ""

    ucs = util.UniformCostSearch(verbose=1)
    ucs.solve(JointSegmentationInsertionProblem(query,bigramCost,possibleFills))
    words=' '.join(ucs.actions)
    return words

if __name__ == "__main__":
    shell.main()
