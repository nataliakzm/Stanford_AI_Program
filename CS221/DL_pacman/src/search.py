"""
Generic search algorithms which are called by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

class Node:
    """AIMA: A node in a search tree. Contains a pointer 
    to the parent (the node that this is a successor of) 
    and to the actual state for this node. Note that if 
    a state is arrived at by two paths, then there are 
    two nodes with the same state.  Also includes the 
    action that got us to this state, and the total 
    path_cost (also known as g) to reach the node.  
    Other functions may add an f and h value; see 
    best_first_graph_search and astar_search for an 
    explanation of how the f and h values are handled. 
    """

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        self.state = state
        self.parent = parent
        self.action = action
        if parent:
            self.path_cost = parent.path_cost + path_cost
            self.depth = parent.depth + 1
        else:
            self.path_cost = path_cost
            self.depth = 0
            
    def __repr__(self):
        return "<Node %s>" % (self.state,)
    
    def nodePath(self):
        "Create a list of nodes from the root to this node."
        x, result = self, [self]
        while x.parent:
            result.append(x.parent)
            x = x.parent
        result.reverse()
        return result
      
    def path(self):
      """
      Create a path of actions from the start to the current state
      """
      actions = []
      currnode = self
      while currnode.parent:
        actions.append(currnode.action)
        currnode = currnode.parent
      actions.reverse()
      return actions
    
    def expand(self, problem):
        "Return a list of nodes reachable from this node. [Fig. 3.8]"
        return [Node(next, self, act, cost)
                for (next, act, cost) in problem.getSuccessors(self.state)]

REVERSE_PUSH = False
def graphSearch(problem, fringe):
    """Search through the successors of a problem to find a goal.
    The argument fringe should be an empty queue. [Fig. 3.18]"""
    startstate = problem.getStartState()
    fringe.push(Node(problem.getStartState()))
    try:
        startstate.__hash__()
        visited = set()
    except:
        visited = list()

    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.isGoalState(node.state): 
            return node.path()
        try:
            inVisited = node.state in visited
        except:
            visited = list(visited)
            inVisited = node.state in visited

        if not inVisited:
            if isinstance(visited, list):
                visited.append(node.state)
            else:
                visited.add(node.state)
            nextNodes = node.expand(problem)
            if REVERSE_PUSH: nextNodes.reverse()
            for nextnode in nextNodes:
                fringe.push(nextnode)
    return None



def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first.
  """
  
  return graphSearch(problem, util.Stack())

def breadthFirstSearch(problem):
  "Search the shallowest nodes in the search tree first."
  
  return graphSearch(problem, util.Queue())
      
def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  
  return graphSearch(problem,
                     util.PriorityQueueWithFunction(
    lambda node: node.path_cost))

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  
  return graphSearch(problem,
                     util.PriorityQueueWithFunction(
    lambda node: node.path_cost + heuristic(node.state, problem)))
    
  
def greedySearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest heuristic first."
  return graphSearch(problem, util.PriorityQueueWithFunction(
    lambda node: heuristic(node.state, problem)))
  
  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch