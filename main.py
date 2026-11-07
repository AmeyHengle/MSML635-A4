import argparse

from config import ExperimentConfig
from src.experiment import agentNames, buildMaze, runExperiment
from src.metrics import summariseResults
from src.storage import saveResults


def parseArguments():
    """
    i let the number of runs, the starting seed, the output folder, the step where the
    shortcut opens and the total number of steps be changed from the command line.
    the defaults all come from config.py and match the assignment. the textbook flag
    is a shortcut for the longer setting from sutton and barto, where the maze changes
    at step 3000 of 6000, and it saves into its own folder inside results.
    """
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Dyna-Q vs Dyna-Q+ in the shortcut maze")
    parser.add_argument("--runs", dest="numRuns", type=int, default=defaults.numRuns, help="number of independent runs per agent")
    parser.add_argument("--seed", dest="baseSeed", type=int, default=defaults.baseSeed, help="seed of the first run")
    parser.add_argument("--output", dest="resultsDir", default=defaults.resultsDir, help="folder to write results into")
    parser.add_argument("--change-at", dest="shortcutStep", type=int, default=defaults.shortcutStep, help="time step at which the shortcut opens")
    parser.add_argument("--steps", dest="totalSteps", type=int, default=defaults.totalSteps, help="total number of time steps per run")
    parser.add_argument("--textbook", action="store_true", help="use the textbook setting: change at 3000 of 6000 steps, saved to results/textbookSetting")
    arguments = parser.parse_args()
    if arguments.textbook:
        arguments.shortcutStep = 3000
        arguments.totalSteps = 6000
        if arguments.resultsDir == defaults.resultsDir:
            arguments.resultsDir = f"{defaults.resultsDir}/textbookSetting"
    return arguments


def printSummary(summary):
    """
    i print the handful of numbers i care about most, so i can check the run looks
    sensible before opening the saved files.
    """
    mazeInfo = summary["maze"]
    print()
    print(f"shortest path before / after the change: {mazeInfo['shortestPathBefore']} / {mazeInfo['shortestPathAfter']} steps")
    for agentName in agentNames:
        agentSummary = summary["agents"][agentName]
        finalReward = agentSummary["finalCumulativeReward"]
        discovery = agentSummary["shortcutDiscovery"]
        finalPhase = agentSummary["finalPhaseBehaviour"]
        stepsAfter = discovery["stepsAfterOpening"]["median"]
        stepsAfterText = "never" if stepsAfter is None else f"median {stepsAfter:.0f} steps after it opened"
        print(
            f"{agentName:>10}: final reward {finalReward['mean']:.1f} ± {finalReward['std']:.1f}, "
            f"found shortcut in {discovery['runsFound']}/{summary['config']['numRuns']} runs ({stepsAfterText}), "
            f"final-phase episode length {finalPhase['meanEpisodeLength']['mean']:.1f}"
        )


def main():
    """
    i build the config, run both agents, summarise the runs and save everything into
    the results folder.
    """
    arguments = parseArguments()
    config = ExperimentConfig(
        numRuns=arguments.numRuns,
        baseSeed=arguments.baseSeed,
        resultsDir=arguments.resultsDir,
        shortcutStep=arguments.shortcutStep,
        totalSteps=arguments.totalSteps,
    )
    maze = buildMaze(config)
    results = runExperiment(config)
    summary = summariseResults(results, config, maze)
    savedFiles = saveResults(results, summary, config, maze)
    printSummary(summary)
    print()
    print("saved:")
    for path in savedFiles:
        print(f"  {path}")


if __name__ == "__main__":
    main()
