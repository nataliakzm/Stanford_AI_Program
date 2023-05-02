# multiAgents.py
# --------------
# The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
  """
  
  def __init__(self):
    self.lastPositions = []
    self.dc = None
  

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    
    myPos = gameState.getPacmanState().getPosition()
    self.lastPositions.append(myPos)
    if len(self.lastPositions) > 20:
      self.lastPositions.pop(0)
    
    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    

    if self.dc is None:
      self.dc = DistanceCalculator(currentGameState.data.layout)

    repeatPenalty = 0
    numRepeats = sum([1 for x in self.lastPositions if x == newPos])
    if numRepeats > 2:
      repeatPenalty = 0.5

    foodPositions = oldFood.asList()
    foodDistances = [self.dc.getDistance(newPos, foodPosition) for foodPosition in foodPositions]
    closestFood = min(foodDistances)+1.0

    ghostStates = successorGameState.getGhostStates()
    ghostPositions = [ghostState.getPosition() for ghostState in ghostStates if ghostState.scaredTimer < 5]
    ghostDistances = [self.dc.getDistance(newPos, ghostPosition) for ghostPosition in ghostPositions]
    ghostDistances.append(1000)
    closestGhost = min(ghostDistances)+1.0

    scaredGhostPositions = [ghostState.getPosition() for ghostState in ghostStates if ghostState.scaredTimer >= 5]
    scaredGhostDistances = [self.dc.getDistance(newPos, ghostPosition) for ghostPosition in scaredGhostPositions]
    scaredGhostDistances.append(1000)
    closestScaredGhost = min(scaredGhostDistances)+1.0
    scaredCount = len(scaredGhostDistances)

    dangerZone = 0
    if newPos in [ghost.start.getPosition() for ghost in ghostStates]: dangerZone = 1000

    return 1.0/closestFood - 10.0/closestGhost/closestGhost + 10.0/closestScaredGhost + 3*scaredCount - repeatPenalty - dangerZone + successorGameState.getScore()
    return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    Some common elements to all of multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Minimax agent
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    
    return self.agent.getAction(gameState)

  def __init__(self, **args):
    self.agent = StaffMultiAgentSearchAgent(**args)
    

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Minimax agent with alpha-beta pruning
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    return self.agent.getAction(gameState)

  def __init__(self, **args):
    args['alphabeta'] = True
    self.agent = StaffMultiAgentSearchAgent(**args)
    

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Expectimax agent
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    return self.agent.getAction(gameState)

  def __init__(self, **args):
    args['expectimax'] = True
    self.agent = StaffMultiAgentSearchAgent(**args)
    

def betterEvaluationFunction(currentGameState):
  """
    Extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function.
  """
  
  walls = currentGameState.getWalls()
  if walls not in DISTANCE_CALCULATORS:
    DISTANCE_CALCULATORS[walls] = DistanceCalculator(currentGameState.data.layout)
  return staffEvaluationFunction(currentGameState, DISTANCE_CALCULATORS[walls])
  
# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action. 
      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    return self.agent.getAction(gameState)

  def __init__(self, **args):
    args['alphabeta'] = True
    args['depth'] = 3
    self.agent = StaffMultiAgentSearchAgent(**args)
    

DISTANCE_CALCULATORS = {}

def staffEvaluationFunction(state, distanceCalculator):
  newPos = state.getPacmanState().configuration.getPosition()
  food = state.getFood()
  foodPositions = food.asList()
  foodDistances = [distanceCalculator.getDistance(newPos, foodPosition) for foodPosition in foodPositions]
  foodDistances.append(1000)
  closestFood = min(foodDistances)+2.0
  ghostPositions = [ghostState.configuration.getPosition() for ghostState in state.getGhostStates() if ghostState.scaredTimer < 5]
  ghostDistances = [distanceCalculator.getDistance(newPos, ghostPosition) for ghostPosition in ghostPositions]
  ghostDistances.append(1000)
  closestGhost = min(ghostDistances)+1.0
  scaredGhostPositions = [ghostState.configuration.getPosition() for ghostState in state.getGhostStates() if ghostState.scaredTimer >= 5]
  scaredGhostDistances = [distanceCalculator.getDistance(newPos, ghostPosition) for ghostPosition in scaredGhostPositions]
  scaredGhostDistances.append(1000)
  closestScaredGhost = min(scaredGhostDistances)+1.0
  closestGhostPenalty = 1.0/closestGhost**2
  if closestGhostPenalty <= 1.0/25: closestGhostPenalty = 0
  numScared = sum([1 for ghostState in state.getGhostStates() if ghostState.scaredTimer >= 1])
  features = [1.0/closestFood, closestGhostPenalty, 1.0/closestScaredGhost, state.getScore(), numScared]
  weights = [10, -300, 200, .9, 1]
  return sum([feature * weight for feature, weight in zip(features, weights)])

class ManhattanDistanceCalculator:
  def getDistance(self, pos1, pos2):
    return manhattanDistance(pos1, pos2)

class DistanceCalculator:
  """
  The agent should create and store a distance calculator once at initialization time
  and call the getDistance function as necessary.  The remaining functions can be
  ignored.
  """
  def __init__(self, layout, default = 10000):
    """
    Initialize with DistanceCalculator(layout).  Changing default is unnecessary.
    """
    print("Calculating position distances...", end=' ')
    self._distances = {}
    self.default = default
    self.calculateDistances(layout)
    print("done.")

  def getDistance(self, pos1, pos2):
    if self.isInt(pos1) and self.isInt(pos2):
      return self.getDistanceOnGrid(pos1, pos2)
    pos1Grids = self.getGrids2D(pos1)
    pos2Grids = self.getGrids2D(pos2)
    bestDistance = self.default
    for pos1Snap, snap1Distance in pos1Grids:
      for pos2Snap, snap2Distance in pos2Grids:
        gridDistance = self.getDistanceOnGrid(pos1Snap, pos2Snap)
        distance = gridDistance + snap1Distance + snap2Distance
        if bestDistance > distance:
          bestDistance = distance
    return bestDistance

  def getDistanceOnGrid(self, pos1, pos2):
    key = (pos1, pos2)
    if key in self._distances:
      return self._distances[key]
    return self.default

  def isInt(self, pos):
    x, y = pos
    return x == int(x) and y == int(y)


  def getGrids2D(self, pos):
    grids = []
    for x, xDistance in self.getGrids1D(pos[0]):
      for y, yDistance in self.getGrids1D(pos[1]):
        grids.append(((x, y), xDistance + yDistance))
    return grids

  def getGrids1D(self, x):
    intX = int(x)
    if x == int(x):
      return [(x, 0)]
    return [(intX, x-intX), (intX+1, intX+1-x)]

  def manhattanDistance(self, x, y ):
    return abs( x[0] - y[0] ) + abs( x[1] - y[1] )

  def calculateDistances(self, layout):
    allNodes = layout.walls.asList(False)
    remainingNodes = allNodes[:]
    for node in allNodes:
      self._distances[(node, node)] = 0.0
      for otherNode in allNodes:
        if self.manhattanDistance(node, otherNode) == 1.0:
          self._distances[(node, otherNode)] = 1.0
    while len(remainingNodes) > 0:
      node = remainingNodes.pop()
      for node1 in allNodes:
        dist1 = self.getDistanceOnGrid(node1, node)
        for node2 in allNodes:
          oldDist = self.getDistanceOnGrid(node1, node2)
          if dist1 > oldDist:
            continue
          dist2 = self.getDistanceOnGrid(node2, node)
          newDist = dist1 + dist2
          if newDist < oldDist:
            self._distances[(node1, node2)] = newDist
            self._distances[(node2, node1)] = newDist

class StaffMultiAgentSearchAgent(MultiAgentSearchAgent):
  def __init__(self, evalFn='scoreEvaluationFunction', depth='2', expectimax=False, alphabeta=False, distancePrune='false', keepStop='false', usePartialPlyBug='false', verbose='false'):
    MultiAgentSearchAgent.__init__(self, evalFn, depth)
    self.expectimax = expectimax
    self.alphabeta = alphabeta
    self.distancePrune = distancePrune.lower() == 'true'
    self.keepStop = keepStop.lower() == 'true'
    self.usePartialPlyBug = usePartialPlyBug.lower() == 'true'
    self.resetStats()
    self.verbose = verbose.lower() == 'true'

  def getAction(self, state):
    self.resetStats()
    actions, score = self.getBestPacmanActions(state)
    if self.verbose:
      print('Best actions:', actions)
      print('Approximate score:', score)
    self.printStats()
    return actions[0]

  def getBestPacmanActions(self, state):
    return self.getBestActionsAndScore(state, 0, self.depth + 1, -100000, 100000)

  def resetStats(self):
    self._statesVisited = 0
    self._statesEvaluated = 0
    self._branchesPruned = 0
    self._agentsSkipped = 0
    self._states = {}
    self._duplicateStatesEvaluated = 0

  def printStats(self):
    if self.verbose:
      print("States visted:\t", self._statesVisited)
      print("States evaled:\t", self._statesEvaluated)
      print("Branch prunes:\t", self._branchesPruned)
      print("Agents skipped:\t", self._agentsSkipped)
      print("Branching factor:\t", self._statesEvaluated ** (1.0 / self.depth))
      print("Duplicate state rate:\t", (self._duplicateStatesEvaluated+0.0) / self._statesEvaluated)

  def evaluate(self, state):
      if state in self._states:
          self._duplicateStatesEvaluated += 1
      else:
        self._states[state] = 1
      self._statesEvaluated += 1
      return self.evaluationFunction(state)

  def getBestActionsAndScore(self, state, agentIndex, depth, bestMax, bestMin):
    self._statesVisited += 1

    # Skip distant agents
    if self.distancePrune:
      while agentIndex > 0 and self.distanceFromPacman(state, agentIndex) > self.depth * 2:
        self._agentsSkipped += 1
        agentIndex = (agentIndex + 1) % state.getNumAgents()

    if agentIndex == 0: depth -= 1

    if depth == 0 or state.isWin() or state.isLose() or (self.usePartialPlyBug and depth == 1 and agentIndex == state.getNumAgents()-1):
      return None, self.evaluate(state)

    if agentIndex == 0:
      return self.getMax(state, agentIndex, depth, bestMax, bestMin)
    else:
      if not self.expectimax:
        return self.getMin(state, agentIndex, depth, bestMax, bestMin)
      else:
        return self.getExp(state, agentIndex, depth, bestMax, bestMin)

  def distanceFromPacman(self, state, agentIndex):
    pacPos = state.getPacmanPosition()
    ghostPos = state.getGhostPosition(agentIndex)
    return manhattanDistance(pacPos, ghostPos)

  def getMax(self, state, agentIndex, depth, bestMax, bestMin):
    possibleActions = state.getLegalActions(agentIndex)
    bestActions, bestScore = [], -100000

    if not self.keepStop and Directions.STOP in possibleActions:
        possibleActions.remove(Directions.STOP)

    # Code to stop pacman from turning around during search
    #reverse = pacman.Actions.reverseDirection(state.getPacmanState().configuration.direction)
    #if len (possibleActions) > 1 and reverse in possibleActions:
    #  possibleActions.remove(reverse)

    for action in possibleActions:
      successor = state.generateSuccessor(agentIndex, action)
      nextAgent = (agentIndex + 1) % state.getNumAgents()
      nextActions, score = self.getBestActionsAndScore(successor, nextAgent, depth, max(bestScore, bestMax), bestMin)
      if score < bestScore:
        continue
      elif score == bestScore:
        bestActions.append(action)
      else:
        bestActions, bestScore = [action], score
      if self.alphabeta:
        if score > bestMin: # The min agent will never allow this
          self._branchesPruned += 1
          return None, score
    return bestActions, bestScore

  def getMin(self, state, agentIndex, depth, bestMax, bestMin):
    possibleActions = state.getLegalActions(agentIndex)
    bestActions, bestScore = [], 100000
    for action in possibleActions:
      successor = state.generateSuccessor(agentIndex, action)
      nextAgent = (agentIndex + 1) % state.getNumAgents()
      nextActions, score = self.getBestActionsAndScore(successor, nextAgent, depth, bestMax, min(bestMin, bestScore))
      # score += successor.scoreChange
      if score > bestScore:
        continue
      elif score == bestScore:
        bestActions.append(action)
      else:
        bestActions, bestScore = [action], score
      if self.alphabeta:
        if score < bestMax: # The max agent will never allow this
          self._branchesPruned += 1
          return None, score
    return bestActions, bestScore

  i = 0
  def getExp(self, state, agentIndex, depth, bestMax, bestMin):
    possibleActions = state.getLegalActions(agentIndex)
    self.i = self.i + 1
    #if self.i % 200 == 0: print self.i
    avgScore = 0.0
    for action in possibleActions:
      prob = 1.0 / len (possibleActions)
      successor = state.generateSuccessor(agentIndex, action)
      nextAgent = (agentIndex + 1) % state.getNumAgents()
      nextActions, score = self.getBestActionsAndScore(successor, nextAgent, depth, bestMax, bestMin)
      # score += successor.scoreChange
      avgScore += score * prob
    return None, avgScore

class SmartGhost( Agent ):
  def __init__( self, index ):
    self.index = index
    self.minimaxer = StaffMultiAgentSearchAgent(alphabeta=True)
    self.minimaxer.evaluationFunction = self.ghostEvaluationFunction
    self.distanceCalculator = None

  def ghostEvaluationFunction(self, state ):
    if self.distanceCalculator is None:
      walls = state.getWalls()
      self.distanceCalculator = DistanceCalculator(state.data.layout)
    distanceCalculator = self.distanceCalculator
    newPos = state.getPacmanState().configuration.getPosition()
    ghostPositions = [ghostState.configuration.getPosition() for ghostState in state.getGhostStates() if ghostState.scaredTimer < 5]
    ghostDistances = [distanceCalculator.getDistance(newPos, ghostPosition) for ghostPosition in ghostPositions]
    ghostSpaces = [distanceCalculator.getDistance(ga, gb) for ga in ghostPositions for gb in ghostPositions]
    scaredGhostPositions = [ghostState.configuration.getPosition() for ghostState in state.getGhostStates() if ghostState.scaredTimer >= 5]
    scaredGhostDistances = [distanceCalculator.getDistance(newPos, ghostPosition) for ghostPosition in scaredGhostPositions]
    ghostSum = sum(ghostDistances)
    scaredGhostSum = sum(scaredGhostDistances)
    ghostSpacing = sum(ghostSpaces)
    features = [ghostSum, scaredGhostSum, state.getScore(), ghostSpacing]
    weights = [1, -1, 1, -.4]
    return sum([feature * weight for feature, weight in zip(features, weights)])

  def getAction( self, state ):
    actions = self.minimaxer.getBestActionsAndScore(state, self.index, 3, -100000, 100000)[0]
    return random.choice( actions )

  def getDistribution( self, state ):
    actions = self.minimaxer.getBestActionsAndScore(state, self.index, 3, -100000, 100000)[0]
    prob = 1.0 / len( actions )
    return [( prob, action ) for action in actions]

def scoreAndFoodEvaluationFunction(currentGameState):
  """
  Modified eval function for class demo
  """
  newPos = state.getPacmanState().configuration.getPosition()
  food = state.getFood()
  foodPositions = food.asList()
  foodDistances = [distanceCalculator.getDistance(newPos, foodPosition) for foodPosition in foodPositions]
  foodDistances.append(1000)
  closestFood = min(foodDistances)+2.0
  return currentGameState.getScore() - 1.0/closestFood