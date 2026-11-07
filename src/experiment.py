import time

import numpy as np

from src.agents import DynaQAgent, DynaQPlusAgent
from src.maze import ShortcutMaze

agentNames = ("dynaQ", "dynaQPlus")


def buildMaze(config):
    """
    create a fresh maze from the config. every run gets its own maze, so the
    shortcut always starts closed and opens at the same time step for both agents.
    """
    return ShortcutMaze(
        config.numRows,
        config.numCols,
        config.startState,
        config.goalState,
        config.wallsBefore,
        config.wallsAfter,
    )


def buildAgent(agentName, maze, config, rng):
    """
    create either agent with exactly the same hyperparameters. the only extra
    argument dyna-q+ gets is kappa for its exploration bonus.
    """
    sharedArgs = dict(
        numRows=maze.numRows,
        numCols=maze.numCols,
        numActions=maze.numActions,
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon=config.epsilon,
        planningSteps=config.planningSteps,
        rng=rng,
    )
    if agentName == "dynaQ":
        return DynaQAgent(**sharedArgs)
    if agentName == "dynaQPlus":
        return DynaQPlusAgent(**sharedArgs, kappa=config.kappa)
    raise ValueError(f"unknown agent name: {agentName}")


def runSingleTrial(agentName, config, seed):
    """
    run one agent for the full number of time steps. at the shortcut step open
    the shortcut in the maze only, the agent is not told. after the agent reaches the
    goal send it back to the start but keep counting time steps, since the task is
    measured in steps and not episodes. besides the reward at every step, record
    when the agent first crosses the wall through the shortcut from the start's side,
    the length of every episode plus whether that episode used the shortcut, and the
    final q-table so can draw the learned policy later.
    """
    rng = np.random.default_rng(seed)
    maze = buildMaze(config)
    agent = buildAgent(agentName, maze, config, rng)
    stepRewards = np.zeros(config.totalSteps)
    episodes = []
    discoveryStep = None
    episodeStartStep = 0
    episodeUsedShortcut = False
    state = maze.reset()

    for timeStep in range(config.totalSteps):
        if timeStep == config.shortcutStep:
            maze.openShortcut()

        agent.observeState(state, timeStep)
        action = agent.selectAction(state)
        nextState, reward, reachedGoal = maze.step(action)
        agent.learn(state, action, reward, nextState, reachedGoal, timeStep)
        stepRewards[timeStep] = reward

        if maze.isShortcutCrossing(state, nextState):
            episodeUsedShortcut = True
            if discoveryStep is None:
                discoveryStep = timeStep

        if reachedGoal:
            episodes.append(
                {
                    "endStep": timeStep,
                    "length": timeStep - episodeStartStep + 1,
                    "usedShortcut": episodeUsedShortcut,
                }
            )
            episodeStartStep = timeStep + 1
            episodeUsedShortcut = False
            state = maze.reset()
        else:
            state = nextState

    return {
        "seed": seed,
        "stepRewards": stepRewards,
        "cumulativeReward": np.cumsum(stepRewards),
        "discoveryStep": discoveryStep,
        "episodes": episodes,
        "finalQValues": agent.qValues.copy(),
    }


def runExperiment(config, verbose=True):
    """
    repeat the trial for the configured number of runs for both agents. in run i
    both agents use the same seed (the base seed plus i), so they start from
    identical random number streams and
    the only thing that differs between them is the algorithm. print a short
    progress line per run so can see the experiment moving.
    """
    results = {agentName: [] for agentName in agentNames}
    startTime = time.time()
    for runIndex in range(config.numRuns):
        seed = config.baseSeed + runIndex
        for agentName in agentNames:
            results[agentName].append(runSingleTrial(agentName, config, seed))
        if verbose:
            finalRewards = ", ".join(
                f"{agentName}={results[agentName][-1]['cumulativeReward'][-1]:.0f}"
                for agentName in agentNames
            )
            elapsed = time.time() - startTime
            print(f"run {runIndex + 1}/{config.numRuns} (seed {seed}): {finalRewards}  [{elapsed:.1f}s]")
    return results
