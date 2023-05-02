'''
Licensing Information: The Driverless Car project was developed at Stanford, primarily by
Chris Piech (piech@cs.stanford.edu). It was inspired by the Pacman projects.
'''
from engine.const import Const
import util, math, random, collections

# Class: ExactInference
# ---------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using exact updates (correct, but slow times).
class ExactInference(object):
    # Function: Init
    # --------------
    # Constructor that initializes an ExactInference object which has
    # numRows x numCols number of tiles.
    def __init__(self, numRows, numCols):
        self.skipElapse = False
        self.belief = util.Belief(numRows, numCols)
        self.transProb = util.loadTransProb()

    # Observe (update the probabilities based on an observation)
    def observe(self, agentX, agentY, observedDist):
        for row in range(self.belief.numRows):
            for col in range(self.belief.numCols):
                dist = math.sqrt((util.colToX(col) - agentX) ** 2 + (util.rowToY(row) - agentY) ** 2)
                prob_distr = util.pdf(dist, Const.SONAR_STD, observedDist)
                self.belief.setProb(row, col, self.belief.getProb(row, col) * prob_distr)
        self.belief.normalize()

    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    def elapseTime(self):
        if self.skipElapse: return
        newBelief = util.Belief(self.belief.numRows, self.belief.numCols, value=0)
        for oldTile, newTile in self.transProb:
            newBelief.addProb(newTile[0], newTile[1], self.belief.getProb(*oldTile) * self.transProb[(oldTile, newTile)])
        newBelief.normalize()
        self.belief = newBelief        

    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile.
    def getBelief(self):
        return self.belief


# Class: Particle Filter
# ----------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using a set of particles.
class ParticleFilter(object):
    NUM_PARTICLES = 200

    # Function: Init
    # --------------
    # Constructor that initializes an ParticleFilter object which has
    # (numRows x numCols) number of tiles.
    def __init__(self, numRows, numCols):
        self.belief = util.Belief(numRows, numCols)

        # Load the transition probabilities and store them in an integer-valued defaultdict.
        # Use self.transProbDict[oldTile][newTile] to get the probability of transitioning from oldTile to newTile.
        self.transProb = util.loadTransProb()
        self.transProbDict = dict()
        for (oldTile, newTile) in self.transProb:
            if not oldTile in self.transProbDict:
                self.transProbDict[oldTile] = collections.defaultdict(int)
            self.transProbDict[oldTile][newTile] = self.transProb[(oldTile, newTile)]

        # Initialize the particles randomly.
        self.particles = collections.defaultdict(int)
        potentialParticles = list(self.transProbDict.keys())
        for i in range(self.NUM_PARTICLES):
            particleIndex = int(random.random() * len(potentialParticles))
            self.particles[potentialParticles[particleIndex]] += 1

        self.updateBelief()

    # Function: Update Belief
    # ---------------------
    # Updates |self.belief| with the probability that the car is in each tile
    # based on |self.particles|, which is a defaultdict from particle to
    # probability (which should sum to 1).
    def updateBelief(self):
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for tile in self.particles:
            newBelief.setProb(tile[0], tile[1], self.particles[tile])
        newBelief.normalize()
        self.belief = newBelief

    # Function: Observe:
    def observe(self, agentX, agentY, observedDist):
        proposed = collections.defaultdict(float)
        for row, col in self.particles:
            dist = math.sqrt((util.colToX(col) - agentX) ** 2 + (util.rowToY(row) - agentY) ** 2)
            prob_distr = util.pdf(dist, Const.SONAR_STD, observedDist)
            proposed[(row, col)] = self.particles[(row, col)] * prob_distr
        newParticles = collections.defaultdict(int)
        for i in range(self.NUM_PARTICLES):
            particle = util.weightedRandomChoice(proposed)
            newParticles[particle] += 1
        self.particles = newParticles        
        self.updateBelief()

    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    def elapseTime(self):
        newParticles = collections.defaultdict(int)
        for tile, value in self.particles.items():
            if tile in self.transProbDict:
                for _ in range(value):
                    newWeightDict = self.transProbDict[tile]
                    particle = util.weightedRandomChoice(newWeightDict)
                    newParticles[particle] += 1
        self.particles = newParticles
        self.updateBelief()        

    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile.
    def getBelief(self):
        return self.belief
