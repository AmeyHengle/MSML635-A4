import numpy as np

from src.experiment import agentNames


def describe(values):
    """
    summarise a list of numbers with the mean, standard deviation, median, minimum
    and maximum. if the list is empty, for example when an agent never found the
    shortcut, return none for each value instead of crashing.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None, "count": 0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "count": int(values.size),
    }


def rewardRateWindows(totalSteps, windowSize):
    """
    split the run into equal windows of a fixed number of steps. with the default settings
    this gives six windows of 500 steps, and the first one ends exactly when the
    shortcut opens.
    """
    return [(start, min(start + windowSize, totalSteps)) for start in range(0, totalSteps, windowSize)]


def lateEpisodeStats(episodes, fromStep):
    """
    look only at the episodes that finished at or after a given step and return how
    long they took on average and what fraction of them went through the shortcut.
    this tells me which route each agent ends up using.
    """
    lateEpisodes = [episode for episode in episodes if episode["endStep"] >= fromStep]
    if not lateEpisodes:
        return {"episodeCount": 0, "meanEpisodeLength": None, "shortcutEpisodeFraction": None}
    return {
        "episodeCount": len(lateEpisodes),
        "meanEpisodeLength": float(np.mean([episode["length"] for episode in lateEpisodes])),
        "shortcutEpisodeFraction": float(np.mean([episode["usedShortcut"] for episode in lateEpisodes])),
    }


def earlyLearningStats(episodes, shortcutStep):
    """
    measure how far an agent got before the shortcut opened: how long its very first
    episode took and how many episodes it finished before the change. use this to
    check whether the agent had actually settled on the old path when the maze changed.
    """
    firstLength = episodes[0]["length"] if episodes else None
    episodesBefore = sum(1 for episode in episodes if episode["endStep"] < shortcutStep)
    return firstLength, episodesBefore


def summariseAgent(trials, config, windowSize, lateFromStep):
    """
    turn all runs of one agent into the numbers want for the writeup: total reward,
    reward collected before and after the shortcut opened, how much it had learned by
    the change, the reward rate in each window, how often and how quickly the shortcut
    was found, and how the agent was behaving over the final part of the run.
    """
    earlyStats = [earlyLearningStats(trial["episodes"], config.shortcutStep) for trial in trials]
    firstLengths = [firstLength for firstLength, _ in earlyStats if firstLength is not None]
    episodesBefore = [count for _, count in earlyStats]
    cumulative = np.array([trial["cumulativeReward"] for trial in trials])
    stepRewards = np.array([trial["stepRewards"] for trial in trials])
    rewardAtShortcut = stepRewards[:, : config.shortcutStep].sum(axis=1)
    finalReward = cumulative[:, -1]

    windowRates = []
    for startStep, endStep in rewardRateWindows(config.totalSteps, windowSize):
        perRunRates = stepRewards[:, startStep:endStep].sum(axis=1) / (endStep - startStep)
        windowRates.append(
            {
                "startStep": startStep,
                "endStep": endStep,
                "meanRate": float(perRunRates.mean()),
                "stdRate": float(perRunRates.std()),
            }
        )

    discoverySteps = [trial["discoveryStep"] for trial in trials if trial["discoveryStep"] is not None]
    stepsAfterOpening = [step - config.shortcutStep for step in discoverySteps]

    lateStats = [lateEpisodeStats(trial["episodes"], lateFromStep) for trial in trials]
    lateLengths = [stats["meanEpisodeLength"] for stats in lateStats if stats["meanEpisodeLength"] is not None]
    lateFractions = [stats["shortcutEpisodeFraction"] for stats in lateStats if stats["shortcutEpisodeFraction"] is not None]

    return {
        "finalCumulativeReward": describe(finalReward),
        "rewardBeforeShortcut": describe(rewardAtShortcut),
        "rewardAfterShortcut": describe(finalReward - rewardAtShortcut),
        "learningBeforeShortcut": {
            "firstEpisodeLength": describe(firstLengths),
            "episodesFinishedBeforeShortcut": describe(episodesBefore),
            "runsWithNoEpisodeFinishedBeforeShortcut": int(sum(1 for count in episodesBefore if count == 0)),
        },
        "rewardRateByWindow": windowRates,
        "shortcutDiscovery": {
            "runsFound": len(discoverySteps),
            "fractionOfRunsFound": len(discoverySteps) / len(trials),
            "stepsAfterOpening": describe(stepsAfterOpening),
        },
        "finalPhaseBehaviour": {
            "fromStep": lateFromStep,
            "meanEpisodeLength": describe(lateLengths),
            "shortcutEpisodeFraction": describe(lateFractions),
        },
    }


def summariseResults(results, config, maze, windowSize=500, lateSteps=1000):
    """
    build the full summary that gets saved as json. besides each agent's numbers,
    store the shortest path lengths from the maze (the best any agent could do in
    each phase) and a paired comparison of the final rewards, which is fair because
    both agents used the same seed in every run.
    """
    lateFromStep = config.totalSteps - lateSteps
    shortestBefore = maze.shortestPathLength(maze.wallsBefore)
    shortestAfter = maze.shortestPathLength(maze.wallsAfter)

    agentSummaries = {}
    for agentName in agentNames:
        agentSummaries[agentName] = summariseAgent(results[agentName], config, windowSize, lateFromStep)

    finalDynaQ = np.array([trial["cumulativeReward"][-1] for trial in results["dynaQ"]])
    finalDynaQPlus = np.array([trial["cumulativeReward"][-1] for trial in results["dynaQPlus"]])
    pairedDifference = finalDynaQPlus - finalDynaQ

    return {
        "config": config.toDict(),
        "maze": {
            "shortcutCells": [list(cell) for cell in sorted(maze.shortcutCells)],
            "shortestPathBefore": shortestBefore,
            "shortestPathAfter": shortestAfter,
            "bestRewardRateBefore": 1.0 / shortestBefore,
            "bestRewardRateAfter": 1.0 / shortestAfter,
        },
        "agents": agentSummaries,
        "comparison": {
            "finalRewardDifferenceDynaQPlusMinusDynaQ": describe(pairedDifference),
            "runsWhereDynaQPlusAhead": int((pairedDifference > 0).sum()),
            "runsTied": int((pairedDifference == 0).sum()),
            "runsWhereDynaQAhead": int((pairedDifference < 0).sum()),
        },
    }


def perRunRows(results, config, lateSteps=1000):
    """
    flatten the results into one row per agent per run, so can open them as a
    table and look at individual runs instead of only the averages.
    """
    lateFromStep = config.totalSteps - lateSteps
    rows = []
    for agentName in agentNames:
        for runIndex, trial in enumerate(results[agentName]):
            lateStats = lateEpisodeStats(trial["episodes"], lateFromStep)
            firstLength, episodesBefore = earlyLearningStats(trial["episodes"], config.shortcutStep)
            discoveryStep = trial["discoveryStep"]
            rows.append(
                {
                    "agent": agentName,
                    "run": runIndex,
                    "seed": trial["seed"],
                    "finalCumulativeReward": float(trial["cumulativeReward"][-1]),
                    "rewardBeforeShortcut": float(trial["stepRewards"][: config.shortcutStep].sum()),
                    "firstEpisodeLength": "" if firstLength is None else firstLength,
                    "episodesBeforeShortcut": episodesBefore,
                    "shortcutFoundAtStep": "" if discoveryStep is None else discoveryStep,
                    "stepsAfterOpening": "" if discoveryStep is None else discoveryStep - config.shortcutStep,
                    "finalPhaseEpisodes": lateStats["episodeCount"],
                    "finalPhaseMeanEpisodeLength": lateStats["meanEpisodeLength"],
                    "finalPhaseShortcutFraction": lateStats["shortcutEpisodeFraction"],
                }
            )
    return rows
