import numpy as np

from config import ExperimentConfig
from src.agents import DynaQAgent, DynaQPlusAgent
from src.experiment import buildAgent, buildMaze, runSingleTrial


def makeAgent(agentName, planningSteps=10, seed=0):
    """
    i build a maze and an agent from the default config so every test starts from
    the same setup as the real experiment.
    """
    config = ExperimentConfig(planningSteps=planningSteps)
    maze = buildMaze(config)
    agent = buildAgent(agentName, maze, config, np.random.default_rng(seed))
    return maze, agent


def testShortestPathLengths():
    """
    i check that the best route is 16 steps through the left gap before the change
    and 10 steps through the shortcut after it.
    """
    maze = buildMaze(ExperimentConfig())
    assert maze.shortestPathLength(maze.wallsBefore) == 16
    assert maze.shortestPathLength(maze.wallsAfter) == 10


def testWallsAndEdgesKeepAgentInPlace():
    """
    i check that moving into a wall or off the grid leaves the agent where it was.
    """
    maze = buildMaze(ExperimentConfig())
    up, down, left, right = range(4)
    assert maze.nextCell((4, 1), up) == (4, 1)
    assert maze.nextCell((5, 3), down) == (5, 3)
    assert maze.nextCell((0, 0), left) == (0, 0)
    assert maze.nextCell((4, 0), up) == (3, 0)


def testShortcutOpensOnlyColumnEight():
    """
    i check that opening the shortcut lets the agent move up from (4, 8) but that
    the rest of the wall is still there.
    """
    maze = buildMaze(ExperimentConfig())
    up = 0
    assert maze.nextCell((4, 8), up) == (4, 8)
    maze.openShortcut()
    assert maze.nextCell((4, 8), up) == (3, 8)
    assert maze.nextCell((4, 7), up) == (4, 7)
    assert maze.shortcutCells == {(3, 8)}


def testShortcutCrossingOnlyCountsFromStartSide():
    """
    i check that stepping up into (3, 8) from below counts as using the shortcut,
    while stepping down into it from above does not.
    """
    maze = buildMaze(ExperimentConfig())
    assert maze.isShortcutCrossing((4, 8), (3, 8))
    assert not maze.isShortcutCrossing((2, 8), (3, 8))


def testGoalGivesRewardAndIsTerminal():
    """
    i check that entering the goal gives a reward of 1 and that the q-learning target
    for that move is just the reward, with no bootstrapping from the goal.
    """
    maze, agent = makeAgent("dynaQ")
    maze.agentState = (1, 8)
    nextState, reward, reachedGoal = maze.step(0)
    assert nextState == (0, 8) and reward == 1.0 and reachedGoal
    agent.qValues[0, 8] = 5.0
    agent.qLearningUpdate((1, 8), 0, reward, nextState, reachedGoal)
    assert agent.qValues[1, 8, 0] == 1.0


def testOnlyDynaQPlusAddsPlaceholders():
    """
    i check that dyna-q+ adds all four actions of a newly visited state to its model
    as zero-reward self-loops, and that plain dyna-q adds nothing.
    """
    _, plainAgent = makeAgent("dynaQ")
    _, plusAgent = makeAgent("dynaQPlus")
    plainAgent.observeState((5, 3), 0)
    plusAgent.observeState((5, 3), 0)
    assert plainAgent.model == {}
    assert len(plusAgent.model) == 4
    for action in range(4):
        assert plusAgent.model[(5, 3, action)] == (0.0, (5, 3), False)


def testRealTransitionOverwritesPlaceholder():
    """
    i check that once an action is really taken, its placeholder is replaced by the
    true outcome and its last-tried time is updated.
    """
    _, plusAgent = makeAgent("dynaQPlus", planningSteps=0)
    plusAgent.observeState((5, 3), 0)
    plusAgent.learn((5, 3), 2, 0.0, (5, 2), False, 7)
    assert plusAgent.model[(5, 3, 2)] == (0.0, (5, 2), False)
    assert plusAgent.lastTried[5, 3, 2] == 7


def testBonusOnlyAppearsInPlanning():
    """
    i check that with planning turned off both agents make exactly the same direct
    update, and that the dyna-q+ planning reward is the model reward plus
    kappa * sqrt(tau).
    """
    _, plainAgent = makeAgent("dynaQ", planningSteps=0)
    _, plusAgent = makeAgent("dynaQPlus", planningSteps=0)
    for agent in (plainAgent, plusAgent):
        agent.qValues[0, 7] = 0.5
        agent.learn((1, 7), 0, 0.0, (0, 7), False, 400)
    assert plainAgent.qValues[1, 7, 0] == plusAgent.qValues[1, 7, 0]
    assert plainAgent.planningReward((1, 7, 1), 0.0, 900) == 0.0
    assert np.isclose(plusAgent.planningReward((1, 7, 1), 0.0, 900), 0.001 * np.sqrt(900))
    assert np.isclose(plusAgent.planningReward((1, 7, 0), 0.0, 900), 0.001 * np.sqrt(500))


def testTrialIsReproducible():
    """
    i check that running the same agent with the same seed twice gives exactly the
    same rewards, so the paired comparison between agents is fair.
    """
    config = ExperimentConfig(totalSteps=600)
    firstTrial = runSingleTrial("dynaQPlus", config, seed=3)
    secondTrial = runSingleTrial("dynaQPlus", config, seed=3)
    assert np.array_equal(firstTrial["stepRewards"], secondTrial["stepRewards"])
