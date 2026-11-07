import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

seriesColors = {"dynaQ": "#2a78d6", "dynaQPlus": "#eb6834"}
seriesLabels = {"dynaQ": "Dyna-Q", "dynaQPlus": "Dyna-Q+"}
surfaceColor = "#fcfcfb"
primaryInk = "#0b0b0b"
secondaryInk = "#52514e"
mutedInk = "#898781"
gridColor = "#e1e0d9"
axisColor = "#c3c2b7"
tintColor = "#f0efec"


def styleAxes(axes):
    """
    keep the chart quiet so the data stands out: light horizontal gridlines, no top
    or right border, and grey tick labels.
    """
    axes.set_facecolor(surfaceColor)
    axes.grid(axis="y", color=gridColor, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(axisColor)
    axes.tick_params(colors=secondaryInk, labelsize=9)


def markShortcut(axes, shortcutStep, showLabel=True):
    """
    draw a dashed vertical line where the shortcut opens, with a small label next to
    it, so every time-based figure marks the change in the same way.
    """
    axes.axvline(shortcutStep, color=mutedInk, linestyle=(0, (4, 3)), linewidth=1.2)
    if showLabel:
        axes.annotate(
            f"shortcut opens\n(t = {shortcutStep})",
            xy=(shortcutStep, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(6, -6),
            textcoords="offset points",
            va="top",
            fontsize=9,
            color=secondaryInk,
        )


def labelLineEnd(axes, xValue, yValue, text):
    """
    write a label just to the right of where a line ends, in normal text colour,
    so the reader can tell the lines apart without going back to the legend.
    """
    axes.annotate(
        text,
        xy=(xValue, yValue),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=10,
        color=primaryInk,
        annotation_clip=False,
    )


def styleLegend(axes, location="upper left", anchor=(0.0, 1.0)):
    """
    add a frameless legend with dark text in the given corner.
    """
    legend = axes.legend(loc=location, bbox_to_anchor=anchor, frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(primaryInk)
    return legend


def addTitles(figure, title, subtitle=None):
    """
    put a bold title in the top left of the figure and an optional grey subtitle
    under it, so all report figures share the same header style. call this after
    the layout is set, line the title up with the left edge of the leftmost plot, and
    place the subtitle a fixed distance in points below the title so they never
    overlap whatever the figure height is.
    """
    leftEdges = []
    for axes in figure.axes:
        axes.apply_aspect()
        leftEdges.append(axes.get_position().x0)
    leftEdge = min(leftEdges) if leftEdges else 0.07
    figure.text(leftEdge, 0.975, title, ha="left", va="top", fontsize=13, color=primaryInk, fontweight="bold")
    if subtitle:
        subtitleTransform = transforms.offset_copy(figure.transFigure, fig=figure, y=-21, units="points")
        figure.text(leftEdge, 0.975, subtitle, ha="left", va="top", fontsize=9.5, color=secondaryInk, transform=subtitleTransform)


def saveFigure(figure, outputPath):
    """
    save a png for quick viewing and a pdf that stays sharp in the report, then
    close the figure so memory does not build up when make many plots.
    """
    pngPath = outputPath.with_suffix(".png")
    pdfPath = outputPath.with_suffix(".pdf")
    figure.savefig(pngPath, facecolor=surfaceColor, dpi=200)
    figure.savefig(pdfPath, facecolor=surfaceColor)
    plt.close(figure)
    return [pngPath, pdfPath]


def drawCumulativeReward(axes, curves, shortcutStep, totalSteps, showBand=True, showLegend=True):
    """
    draw cumulative reward against time step on the given axes. curves maps each
    agent to a (runs x steps) array or a single curve. draw the mean across runs
    and, when there is more than one run, a shaded band of one standard deviation.
    each line gets a label at its end with its final value. put the legend just to
    the right of the shortcut line and below its label, where the curves never go.
    """
    styleAxes(axes)
    timeSteps = np.arange(1, totalSteps + 1)
    for agentName, runCurves in curves.items():
        runCurves = np.atleast_2d(runCurves)
        meanCurve = runCurves.mean(axis=0)
        color = seriesColors[agentName]
        if showBand and runCurves.shape[0] > 1:
            stdCurve = runCurves.std(axis=0)
            axes.fill_between(timeSteps, meanCurve - stdCurve, meanCurve + stdCurve, color=color, alpha=0.15, linewidth=0)
        axes.plot(timeSteps, meanCurve, color=color, linewidth=2, label=seriesLabels[agentName])
        labelLineEnd(axes, timeSteps[-1], meanCurve[-1], f"{seriesLabels[agentName]}  {meanCurve[-1]:.0f}")

    markShortcut(axes, shortcutStep)
    axes.set_xlim(0, totalSteps)
    axes.set_ylim(bottom=0)
    axes.set_xlabel("Time step", color=secondaryInk, fontsize=10)
    axes.set_ylabel("Cumulative reward", color=secondaryInk, fontsize=10)
    if showLegend:
        styleLegend(axes, anchor=(shortcutStep / totalSteps + 0.02, 0.86))


def plotCumulativeReward(curves, config, outputPath, subtitle, showBand=True, title="Dyna-Q vs Dyna-Q+ in the shortcut maze"):
    """
    make the standalone cumulative reward plot the assignment asks for: the mean
    curve of each agent, a band of one standard deviation, a dashed line where the
    shortcut opens, a legend and end labels.
    """
    figure, axes = plt.subplots(figsize=(9, 5.4))
    figure.patch.set_facecolor(surfaceColor)
    drawCumulativeReward(axes, curves, config.shortcutStep, config.totalSteps, showBand=showBand)
    figure.subplots_adjust(left=0.08, right=0.86, top=0.84, bottom=0.11)
    addTitles(figure, title, subtitle)
    return saveFigure(figure, outputPath)
