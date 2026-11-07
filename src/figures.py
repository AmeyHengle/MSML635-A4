import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src.experiment import agentNames, buildMaze
from src.plotting import (
    addTitles,
    axisColor,
    drawCumulativeReward,
    labelLineEnd,
    markShortcut,
    mutedInk,
    primaryInk,
    saveFigure,
    secondaryInk,
    seriesColors,
    seriesLabels,
    styleAxes,
    styleLegend,
    surfaceColor,
    tintColor,
)

wallColor = axisColor


def drawMazeGrid(axes, maze, walls, outlineCells=()):
    """
    draw the maze as a grid of squares with row 0 at the top, matching the
    assignment. walls are filled grey, the start and goal get a light tint and their
    letters, and any cell pass in to outline gets a dashed border so can point out
    the shortcut. also number the rows and columns so cells can be referred to by
    their coordinates.
    """
    axes.set_facecolor(surfaceColor)
    for row in range(maze.numRows):
        for col in range(maze.numCols):
            cell = (row, col)
            if cell in walls:
                fillColor = wallColor
            elif cell in (maze.startState, maze.goalState):
                fillColor = tintColor
            else:
                fillColor = surfaceColor
            axes.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor=fillColor, edgecolor="#e1e0d9", linewidth=1))
    for cell, letter in ((maze.startState, "S"), (maze.goalState, "G")):
        axes.text(cell[1], cell[0], letter, ha="center", va="center", fontsize=13, fontweight="bold", color=primaryInk, zorder=5)
    for cell in outlineCells:
        axes.add_patch(
            Rectangle((cell[1] - 0.46, cell[0] - 0.46), 0.92, 0.92, fill=False, edgecolor=secondaryInk, linewidth=1.4, linestyle=(0, (3, 2)), zorder=4)
        )
    axes.set_xlim(-0.5, maze.numCols - 0.5)
    axes.set_ylim(maze.numRows - 0.5, -0.5)
    axes.set_aspect("equal")
    axes.set_xticks(range(maze.numCols))
    axes.set_yticks(range(maze.numRows))
    axes.xaxis.tick_top()
    axes.tick_params(length=0, labelsize=8.5, colors=mutedInk)
    for spine in axes.spines.values():
        spine.set_visible(False)


def drawRoute(axes, path, color, lineStyle="-", lineWidth=2.6, zorder=6):
    """
    draw a route through the centres of its cells and put an arrowhead on the last
    step so the direction of travel is clear. pull both ends of the line slightly
    away from the start and goal centres so the letters stay readable.
    """
    if len(path) < 2:
        return
    points = np.array([[cell[1], cell[0]] for cell in path], dtype=float)
    points[0] += 0.3 * (points[1] - points[0])
    points[-1] -= 0.32 * (points[-1] - points[-2])
    axes.plot(points[:-1, 0], points[:-1, 1], color=color, linewidth=lineWidth, linestyle=lineStyle, solid_capstyle="round", zorder=zorder)
    axes.annotate(
        "",
        xy=(points[-1, 0], points[-1, 1]),
        xytext=(points[-2, 0], points[-2, 1]),
        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=lineWidth, linestyle="-", mutation_scale=16, shrinkA=0, shrinkB=0),
        zorder=zorder,
    )


def plotMazeLayouts(config, outputPath):
    """
    figure 1. draw the maze before and after the change side by side. on each one
    draw the shortest route (the one with the fewest turns among the shortest), and
    on the second maze also show the old route as a dashed line so the reader can
    compare the 16-step and 10-step routes directly.
    """
    maze = buildMaze(config)
    oldRoute = maze.shortestPath(maze.wallsBefore)
    newRoute = maze.shortestPath(maze.wallsAfter)
    shortcutCells = sorted(maze.shortcutCells)

    figure, (axesBefore, axesAfter) = plt.subplots(1, 2, figsize=(11, 4.6))
    figure.patch.set_facecolor(surfaceColor)

    drawMazeGrid(axesBefore, maze, maze.wallsBefore)
    drawRoute(axesBefore, oldRoute, primaryInk)
    axesBefore.set_title(
        f"(a) before step {config.shortcutStep}: only the left gap is open\nshortest route: {len(oldRoute) - 1} steps",
        loc="left", fontsize=10, color=primaryInk, pad=22,
    )

    drawMazeGrid(axesAfter, maze, maze.wallsAfter, outlineCells=shortcutCells)
    drawRoute(axesAfter, oldRoute, mutedInk, lineStyle=(0, (3, 2)), lineWidth=1.6, zorder=5)
    drawRoute(axesAfter, newRoute, primaryInk)
    axesAfter.set_title(
        f"(b) from step {config.shortcutStep}: shortcut at {shortcutCells[0]} opens\nshortest route: {len(newRoute) - 1} steps (old route dashed)",
        loc="left", fontsize=10, color=primaryInk, pad=22,
    )

    figure.subplots_adjust(left=0.05, right=0.98, top=0.7, bottom=0.04, wspace=0.12)
    addTitles(figure, "The changing maze", "Grey cells are walls. Rows and columns are 0-indexed, with row 0 at the top.")
    return saveFigure(figure, outputPath)


def plotCumulativeComparison(curves, config, runIndex, seed, outputPath):
    """
    figure 2. put the averaged cumulative reward next to a single run. the average
    shows the overall result, and the single run shows what one agent actually does,
    including the change in slope when dyna-q+ switches to the shortcut.
    """
    figure, (axesMean, axesSingle) = plt.subplots(1, 2, figsize=(13, 5.0))
    figure.patch.set_facecolor(surfaceColor)
    numRuns = next(iter(curves.values())).shape[0]

    drawCumulativeReward(axesMean, curves, config.shortcutStep, config.totalSteps)
    axesMean.set_title(f"(a) mean of {numRuns} runs, band = ±1 standard deviation", loc="left", fontsize=10, color=primaryInk, pad=10)

    singleCurves = {agentName: runCurves[runIndex] for agentName, runCurves in curves.items()}
    drawCumulativeReward(axesSingle, singleCurves, config.shortcutStep, config.totalSteps, showBand=False, showLegend=False)
    axesSingle.set_title(f"(b) a single run (seed {seed})", loc="left", fontsize=10, color=primaryInk, pad=10)

    figure.subplots_adjust(left=0.06, right=0.9, top=0.84, bottom=0.11, wspace=0.32)
    addTitles(figure, "Cumulative reward over time")
    return saveFigure(figure, outputPath)


def shortcutShareByBin(episodes, agentName, config, binSize):
    """
    split the time after the shortcut opens into bins and, for each run, work out
    what share of the episodes that ended in a bin went through the shortcut. then
    average those shares over the runs that finished at least one episode in that bin,
    so every run counts equally no matter how many episodes it completed.
    """
    binEdges = np.arange(config.shortcutStep, config.totalSteps + binSize, binSize)
    runIds = sorted({int(row["run"]) for row in episodes if row["agent"] == agentName})
    shares = np.full((len(runIds), len(binEdges) - 1), np.nan)
    for runPosition, runId in enumerate(runIds):
        runEpisodes = [row for row in episodes if row["agent"] == agentName and int(row["run"]) == runId]
        endSteps = np.array([int(row["endStep"]) for row in runEpisodes])
        usedShortcut = np.array([int(row["usedShortcut"]) for row in runEpisodes])
        for binIndex in range(len(binEdges) - 1):
            inBin = (endSteps >= binEdges[binIndex]) & (endSteps < binEdges[binIndex + 1])
            if inBin.any():
                shares[runPosition, binIndex] = usedShortcut[inBin].mean()
    binCentres = (binEdges[:-1] + binEdges[1:]) / 2
    return binCentres, np.nanmean(shares, axis=0)


def plotShortcutUsage(episodes, config, numRuns, outputPath, binSize=250):
    """
    figure 3. plot the share of episodes that went through the shortcut over time
    for both agents. this shows directly which route each agent is using, which is
    the behaviour behind the slopes in the cumulative reward plot. shade the time
    before the change because the shortcut is still a wall there.
    """
    figure, axes = plt.subplots(figsize=(9, 5.2))
    figure.patch.set_facecolor(surfaceColor)
    styleAxes(axes)
    axes.axvspan(0, config.shortcutStep, color=tintColor, linewidth=0)
    axes.text(config.shortcutStep / 2, 50, "shortcut\nnot open yet", ha="center", va="center", fontsize=9, color=mutedInk)

    for agentName in agentNames:
        binCentres, meanShare = shortcutShareByBin(episodes, agentName, config, binSize)
        axes.plot(binCentres, meanShare * 100, color=seriesColors[agentName], linewidth=2, marker="o", markersize=5, label=seriesLabels[agentName])
        lastIndex = np.flatnonzero(~np.isnan(meanShare))[-1]
        labelLineEnd(axes, binCentres[lastIndex], meanShare[lastIndex] * 100, f"{seriesLabels[agentName]}  {meanShare[lastIndex] * 100:.0f}%")

    markShortcut(axes, config.shortcutStep, showLabel=False)
    axes.set_xlim(0, config.totalSteps)
    axes.set_ylim(0, 100)
    axes.set_xlabel("Time step (episodes grouped by the step they ended)", color=secondaryInk, fontsize=10)
    axes.set_ylabel("Episodes that used the shortcut (%)", color=secondaryInk, fontsize=10)
    styleLegend(axes, location="upper left", anchor=(0.2, 1.0))
    figure.subplots_adjust(left=0.09, right=0.85, top=0.84, bottom=0.12)
    addTitles(figure, "Which route the agents use after the change", f"Mean over {numRuns} runs, {binSize}-step bins")
    return saveFigure(figure, outputPath)


def discoveryCurve(stepsAfterOpening, numRuns, horizon):
    """
    turn a list of discovery times into a step curve of the percentage of runs that
    have found the shortcut by each point in time. runs that never found it are left
    out of the list, so the curve tops out below 100% for them.
    """
    sortedSteps = np.sort(np.asarray(stepsAfterOpening, dtype=float))
    xValues = np.concatenate([[0], sortedSteps, [horizon]])
    yValues = np.concatenate([[0], np.arange(1, len(sortedSteps) + 1), [len(sortedSteps)]]) / numRuns * 100
    return xValues, yValues


def plotShortcutDiscovery(perRun, config, outputPath):
    """
    figure 4. plot how quickly each agent first crossed the shortcut, split by how
    far it had got before the change. panel (a) holds runs where the agent had
    already reached the goal at least once, so it had a learned route to stick to.
    panel (b) holds runs where it was still on its very first episode, so it was
    effectively still exploring when the wall opened. the split shows that most of
    dyna-q's discoveries happen in the second group.
    """
    horizon = config.totalSteps - config.shortcutStep
    groups = (
        ("(a) agent had already reached the goal before the change", lambda row: int(row["episodesBeforeShortcut"]) >= 1),
        ("(b) agent was still on its first episode when the shortcut opened", lambda row: int(row["episodesBeforeShortcut"]) == 0),
    )
    figure, panels = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    figure.patch.set_facecolor(surfaceColor)

    for axes, (panelTitle, belongsToGroup) in zip(panels, groups):
        styleAxes(axes)
        for agentName in agentNames:
            groupRows = [row for row in perRun if row["agent"] == agentName and belongsToGroup(row)]
            if not groupRows:
                continue
            foundSteps = [int(row["stepsAfterOpening"]) for row in groupRows if row["stepsAfterOpening"] != ""]
            xValues, yValues = discoveryCurve(foundSteps, len(groupRows), horizon)
            axes.plot(
                xValues, yValues, drawstyle="steps-post", color=seriesColors[agentName], linewidth=2,
                label=f"{seriesLabels[agentName]} (n = {len(groupRows)} runs)",
            )
            labelLineEnd(axes, horizon, yValues[-1], f"{len(foundSteps)}/{len(groupRows)}")
        axes.set_xlim(0, horizon)
        axes.set_ylim(0, 105)
        axes.set_xlabel("Steps since the shortcut opened", color=secondaryInk, fontsize=10)
        axes.set_title(panelTitle, loc="left", fontsize=10, color=primaryInk, pad=10)
        styleLegend(axes, location="center right", anchor=(1.0, 0.45))
    panels[0].set_ylabel("Runs that have found the shortcut (%)", color=secondaryInk, fontsize=10)

    figure.subplots_adjust(left=0.06, right=0.94, top=0.8, bottom=0.11, wspace=0.18)
    addTitles(figure, "How quickly each agent found the shortcut", "A run counts as having found it the first time the agent moves up from (4, 8) into (3, 8)")
    return saveFigure(figure, outputPath)


def greedyPath(maze, qValues, walls, maxSteps=40):
    """
    follow the greedy action from the start, using the final q-table and the final
    wall layout, until the agent reaches the goal, revisits a cell or runs out of
    steps. this is the route the agent would take if it stopped exploring.
    """
    cell = maze.startState
    path = [cell]
    for _ in range(maxSteps):
        action = int(np.argmax(qValues[cell[0], cell[1]]))
        nextCell = maze.nextCell(cell, action, walls)
        path.append(nextCell)
        if nextCell == maze.goalState or nextCell in path[:-1]:
            break
        cell = nextCell
    return path


def describeRoute(maze, path):
    """
    turn a greedy path into a short description: how many steps it takes and
    whether it goes through the shortcut or the left gap, or that it never reaches
    the goal.
    """
    if path[-1] != maze.goalState:
        return "greedy route does not reach the goal"
    usesShortcut = any(maze.isShortcutCrossing(path[index], path[index + 1]) for index in range(len(path) - 1))
    routeName = "through the shortcut" if usesShortcut else "through the left gap"
    return f"greedy route: {len(path) - 1} steps {routeName}"


def plotPolicyMaps(finalQValues, config, runIndex, seed, outputPath):
    """
    figure 5. for one run draw the greedy action in every cell of the final maze as
    a small arrow, and highlight the greedy route from the start in the agent's colour.
    this shows what each agent believes at the end: dyna-q's values still lead to the
    old gap because its model says (4, 8) up hits a wall, while dyna-q+ has gone back,
    tested that action and learned the shortcut. cells where every action has the same
    value get a dot instead of an arrow.
    """
    maze = buildMaze(config)
    walls = maze.wallsAfter
    figure, panels = plt.subplots(1, 2, figsize=(11, 4.9))
    figure.patch.set_facecolor(surfaceColor)

    for axes, agentName in zip(panels, agentNames):
        qValues = finalQValues[agentName][runIndex]
        drawMazeGrid(axes, maze, walls, outlineCells=sorted(maze.shortcutCells))
        for row in range(maze.numRows):
            for col in range(maze.numCols):
                cell = (row, col)
                if cell in walls or cell == maze.goalState:
                    continue
                actionValues = qValues[row, col]
                if np.allclose(actionValues, actionValues[0]):
                    axes.plot(col, row, marker="o", markersize=2.5, color=mutedInk, zorder=3)
                    continue
                rowMove, colMove = maze.actionMoves[int(np.argmax(actionValues))]
                axes.annotate(
                    "",
                    xy=(col + 0.3 * colMove, row + 0.3 * rowMove),
                    xytext=(col - 0.3 * colMove, row - 0.3 * rowMove),
                    arrowprops=dict(arrowstyle="-|>", color=mutedInk, linewidth=1.1, mutation_scale=9),
                    zorder=3,
                )
        path = greedyPath(maze, qValues, walls)
        drawRoute(axes, path, seriesColors[agentName], lineWidth=2.4, zorder=6)
        axes.set_title(f"{seriesLabels[agentName]}\n{describeRoute(maze, path)}", loc="left", fontsize=10, color=primaryInk, pad=22)

    figure.subplots_adjust(left=0.05, right=0.98, top=0.7, bottom=0.04, wspace=0.12)
    addTitles(
        figure,
        f"Greedy policy at the end of the run (seed {seed}, step {config.totalSteps})",
        "Arrows show the highest-valued action in each cell, the coloured line is the greedy route from the start",
    )
    return saveFigure(figure, outputPath)
