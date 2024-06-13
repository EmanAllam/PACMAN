# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        #considering food locations
        foodList = newFood.asList()
        if foodList:
            minFoodDist = min([manhattanDistance(newPos, food) for food in foodList]) # Calculate distance to the nearest food
            score += 1 / minFoodDist 
        else :
            score +=1000

        #Avoid ghosts
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            dist_to_ghost = manhattanDistance(ghostPos, newPos)
            if dist_to_ghost <= 2 and ghostState.scaredTimer == 0:
                score -= 1000
            else:
                score += ghostState.scaredTimer/dist_to_ghost

        '''print("newPos:", newPos, "\nnewFood" ,newFood, "\nnewGhostStates" ,newGhostStates,
               "\nnewScaredTimes" ,newScaredTimes, "\nminFoodDist", minFoodDist)'''

        return score
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

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
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(agentIndex, current_depth, gameState):
            #check for win, loss, or maximum depth
            if gameState.isLose() or gameState.isWin() or current_depth == self.depth:
                return self.evaluationFunction(gameState)

            next_agent = (agentIndex + 1) % gameState.getNumAgents()
            if next_agent == 0:  # All agents have moved, increment depth
                next_depth = current_depth + 1
            else:
                next_depth = current_depth

            if agentIndex == 0:  # Pacman's turn (Maximizer)
                scores = []
                for action in gameState.getLegalActions(agentIndex):
                    _,score = minimax(next_agent, next_depth, gameState.generateSuccessor(agentIndex, action))
                    scores.append((score, action))
                return max(scores)[0]

            else:  # Ghosts' turn (Minimizer)
                scores = []
                for action in gameState.getLegalActions(agentIndex):
                    _,score = minimax(next_agent, next_depth, gameState.generateSuccessor(agentIndex, action))
                    scores.append((score, action))
                return min(scores)[0]

        # Start the minimax process from Pacman at depth 0
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            score = minimax(1, 0, gameState.generateSuccessor(0, action))
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def Alpha_Beta_Minimax(agentIndex, current_depth, gameState, alpha, beta):
            if gameState.isLose() or gameState.isWin() or current_depth == self.depth:
                return self.evaluationFunction(gameState)
            next_agent = (agentIndex + 1) % gameState.getNumAgents()
            if next_agent == 0:  # All agents have moved, increment depth
                next_depth = current_depth + 1
            else:
                next_depth = current_depth


            if agentIndex == 0:  # Pacman's turn (Maximizer)
                bestVal = -float('inf') 
                for action in gameState.getLegalActions(agentIndex):
                    score = Alpha_Beta_Minimax(next_agent, next_depth, gameState.generateSuccessor(agentIndex, action), alpha, beta)
                    bestVal = max( bestVal, score) 
                    alpha = max( alpha, bestVal)
                    if beta <= alpha:
                        break
                return bestVal

            else:  # Ghosts' turn (Minimizer)
                bestVal = float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    score = Alpha_Beta_Minimax(next_agent, next_depth, gameState.generateSuccessor(agentIndex, action), alpha, beta)
                    bestVal = min( bestVal, score) 
                    beta = min( beta, bestVal)
                    if beta <= alpha:
                        break
                return bestVal

        # Start the minimax process from Pacman (agent 0) at depth 0
        return Alpha_Beta_Minimax(0, 0, gameState, -float('inf'), float('inf'))
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectiMax(agentIndex, current_depth, gameState):
            
            #check for win, loss, or maximum depth
            if gameState.isLose() or gameState.isWin() or current_depth == self.depth:
                return self.evaluationFunction(gameState)

            next_agent = (agentIndex + 1) % gameState.getNumAgents()
            if next_agent == 0:  # All agents have moved, increment depth
                next_depth = current_depth + 1
            else:
                next_depth = current_depth

            if agentIndex == 0:  # Pacman's turn (Maximizer)
                scores = []
                for action in gameState.getLegalActions(agentIndex):
                    _,score = expectiMax(next_agent, next_depth, gameState.generateSuccessor(agentIndex, action))
                    scores.append((score, action))
                return max(scores)[0]

            else:  # Ghosts' turn (Minimizer)
                scores = []
                for action in gameState.getLegalActions(agentIndex):
                    _,score = expectiMax(next_agent, next_depth, gameState.generateSuccessor(agentIndex, action))
                    scores.append((score, action))
                return mean(scores)[0] #Returns the average

        # Start the minimax process from Pacman at depth 0
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            score = expectiMax(1, 0, gameState.generateSuccessor(0, action))
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isLose():
        return -float('inf')  # Extremely low score for losing state

    if currentGameState.isWin():
        return float('inf')  # Extremely high score for winning state
    
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    score = scoreEvaluationFunction(currentGameState)

    #considering food locations
    foodList = newFood.asList()
    if foodList:
        minFoodDist = min([manhattanDistance(newPos, food) for food in foodList]) # Calculate distance to the nearest food
        score += 1 / minFoodDist 
    else :
        score +=1000

    #Avoid ghosts
    for ghostState in newGhostStates:
        ghostPos = ghostState.getPosition()
        dist_to_ghost = manhattanDistance(ghostPos, newPos)
        if dist_to_ghost <= 2 and ghostState.scaredTimer == 0:
            score -= 1000
        else:
            score += ghostState.scaredTimer/dist_to_ghost

    # Encourage state with less remaining food
    score -= 3 * len(foodList)
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
