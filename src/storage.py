import csv
import json
from pathlib import Path

import numpy as np

from config import ExperimentConfig
from src.experiment import agentNames
from src.metrics import perRunRows
from src.plotting import plotCumulativeReward


def writeCurvesCsv(results, config, outputPath):
    """
    write the mean and standard deviation of the cumulative reward at every time
    step for both agents. this is the data behind the main plot, so can redraw it
    or quote exact values later without rerunning anything.
    """
    columns = {}
    for agentName in agentNames:
        cumulative = np.array([trial["cumulativeReward"] for trial in results[agentName]])
        columns[f"{agentName}Mean"] = cumulative.mean(axis=0)
        columns[f"{agentName}Std"] = cumulative.std(axis=0)
    with open(outputPath, "w", newline="") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["timeStep", *columns.keys()])
        for stepIndex in range(config.totalSteps):
            writer.writerow([stepIndex + 1, *(f"{values[stepIndex]:.4f}" for values in columns.values())])


def writePerRunCsv(results, config, outputPath):
    """
    write one row per agent per run with that run's key numbers, using the rows
    built in metrics.py.
    """
    rows = perRunRows(results, config)
    with open(outputPath, "w", newline="") as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def writeRawArrays(results, outputPath):
    """
    save the full cumulative reward curve of every run as a compressed numpy file,
    along with the seeds, in case want to make other plots later.
    """
    arrays = {agentName: np.array([trial["cumulativeReward"] for trial in results[agentName]]) for agentName in agentNames}
    arrays["seeds"] = np.array([trial["seed"] for trial in results[agentNames[0]]])
    np.savez_compressed(outputPath, **arrays)


def writeEpisodesCsv(results, outputPath):
    """
    write every completed episode of every run as one row: when it ended, how long
    it took and whether it went through the shortcut. the route usage figure is built
    from this file.
    """
    with open(outputPath, "w", newline="") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["agent", "run", "seed", "endStep", "length", "usedShortcut"])
        for agentName in agentNames:
            for runIndex, trial in enumerate(results[agentName]):
                for episode in trial["episodes"]:
                    writer.writerow(
                        [agentName, runIndex, trial["seed"], episode["endStep"], episode["length"], int(episode["usedShortcut"])]
                    )


def writeFinalQValues(results, outputPath):
    """
    save the q-table each agent ended every run with, shaped (runs, rows, cols,
    actions), so can draw the learned policies without rerunning the experiment.
    """
    arrays = {agentName: np.array([trial["finalQValues"] for trial in results[agentName]]) for agentName in agentNames}
    np.savez_compressed(outputPath, **arrays)


def writeMazeLayouts(maze, outputPath):
    """
    save text drawings of the maze before and after the change, together with the
    shortest path length in each layout, so the setup is documented with the results.
    """
    sections = [
        f"before the shortcut opens (shortest path: {maze.shortestPathLength(maze.wallsBefore)} steps)",
        maze.render(maze.wallsBefore),
        "",
        f"after the shortcut opens (shortest path: {maze.shortestPathLength(maze.wallsAfter)} steps)",
        maze.render(maze.wallsAfter),
        "",
    ]
    Path(outputPath).write_text("\n".join(sections))


def loadSavedResults(resultsDir):
    """
    read back everything the figures need from a results folder: the config and
    summary, the raw reward curves, the per-run and episode tables and the final
    q-tables. this lets me make the report figures without rerunning the experiment.
    """
    resultsDir = Path(resultsDir)
    summary = json.loads((resultsDir / "summary.json").read_text())
    with np.load(resultsDir / "rawCumulativeRewards.npz") as rawFile:
        curves = {agentName: rawFile[agentName] for agentName in agentNames}
        seeds = rawFile["seeds"]
    with np.load(resultsDir / "finalQValues.npz") as qFile:
        finalQValues = {agentName: qFile[agentName] for agentName in agentNames}
    with open(resultsDir / "perRunMetrics.csv", newline="") as csvFile:
        perRun = list(csv.DictReader(csvFile))
    with open(resultsDir / "episodes.csv", newline="") as csvFile:
        episodes = list(csv.DictReader(csvFile))
    return {
        "config": ExperimentConfig.fromDict(summary["config"]),
        "summary": summary,
        "curves": curves,
        "seeds": seeds,
        "finalQValues": finalQValues,
        "perRun": perRun,
        "episodes": episodes,
    }


def saveResults(results, summary, config, maze):
    """
    create the results folder if it does not exist and write everything into it:
    the json summary, the curve, per-run and episode tables, the raw reward curves,
    the final q-tables, the maze layouts and the cumulative reward plot the assignment
    asks for. return the list of files so main can print where everything went.
    """
    resultsDir = Path(config.resultsDir)
    resultsDir.mkdir(parents=True, exist_ok=True)
    savedFiles = []

    summaryPath = resultsDir / "summary.json"
    summaryPath.write_text(json.dumps(summary, indent=2))
    savedFiles.append(summaryPath)

    curvesPath = resultsDir / "cumulativeRewardCurves.csv"
    writeCurvesCsv(results, config, curvesPath)
    savedFiles.append(curvesPath)

    perRunPath = resultsDir / "perRunMetrics.csv"
    writePerRunCsv(results, config, perRunPath)
    savedFiles.append(perRunPath)

    episodesPath = resultsDir / "episodes.csv"
    writeEpisodesCsv(results, episodesPath)
    savedFiles.append(episodesPath)

    rawPath = resultsDir / "rawCumulativeRewards.npz"
    writeRawArrays(results, rawPath)
    savedFiles.append(rawPath)

    qValuesPath = resultsDir / "finalQValues.npz"
    writeFinalQValues(results, qValuesPath)
    savedFiles.append(qValuesPath)

    layoutPath = resultsDir / "mazeLayouts.txt"
    writeMazeLayouts(maze, layoutPath)
    savedFiles.append(layoutPath)

    allCurves = {agentName: np.array([trial["cumulativeReward"] for trial in results[agentName]]) for agentName in agentNames}
    runWord = "run" if config.numRuns == 1 else "runs"
    savedFiles += plotCumulativeReward(
        allCurves,
        config,
        resultsDir / "cumulativeReward",
        subtitle=f"Mean of {config.numRuns} {runWord}, shaded band = ±1 standard deviation",
    )
    return savedFiles
