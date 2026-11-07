import argparse
from pathlib import Path

from src.figures import (
    plotCumulativeComparison,
    plotMazeLayouts,
    plotPolicyMaps,
    plotShortcutDiscovery,
    plotShortcutUsage,
)
from src.plotting import plotCumulativeReward
from src.storage import loadSavedResults


def parseArguments():
    """
    i let the folders and the run used for the single-run panels be changed from the
    command line. by default i read the assignment results from the results folder,
    the textbook results from its own subfolder inside it, and write the figures to
    a figures subfolder.
    """
    parser = argparse.ArgumentParser(description="make the report figures from saved results")
    parser.add_argument("--results", dest="resultsDir", default="results", help="folder with the assignment-setting results")
    parser.add_argument("--textbook", dest="textbookDir", default="results/textbookSetting", help="folder with the textbook-setting results")
    parser.add_argument("--output", dest="figuresDir", default="results/figures", help="folder to write the figures into")
    parser.add_argument("--run", dest="runIndex", type=int, default=0, help="which run to use for the single-run panel and the policy maps")
    return parser.parse_args()


def main():
    """
    i load the saved results and make the six report figures in order. the textbook
    figure is skipped with a message if that experiment has not been run yet, so the
    other five can still be made.
    """
    arguments = parseArguments()
    figuresDir = Path(arguments.figuresDir)
    figuresDir.mkdir(parents=True, exist_ok=True)

    saved = loadSavedResults(arguments.resultsDir)
    config = saved["config"]
    seed = int(saved["seeds"][arguments.runIndex])
    numRuns = len(saved["seeds"])
    savedFiles = []

    savedFiles += plotMazeLayouts(config, figuresDir / "fig1MazeLayouts")
    savedFiles += plotCumulativeComparison(saved["curves"], config, arguments.runIndex, seed, figuresDir / "fig2CumulativeReward")
    savedFiles += plotShortcutUsage(saved["episodes"], config, numRuns, figuresDir / "fig3ShortcutUsage")
    savedFiles += plotShortcutDiscovery(saved["perRun"], config, figuresDir / "fig4ShortcutDiscovery")
    savedFiles += plotPolicyMaps(saved["finalQValues"], config, arguments.runIndex, seed, figuresDir / "fig5PolicyMaps")

    textbookDir = Path(arguments.textbookDir)
    if (textbookDir / "summary.json").exists():
        textbook = loadSavedResults(textbookDir)
        textbookConfig = textbook["config"]
        savedFiles += plotCumulativeReward(
            textbook["curves"],
            textbookConfig,
            figuresDir / "fig6TextbookSetting",
            subtitle=f"Mean of {len(textbook['seeds'])} runs, band = ±1 standard deviation",
            title=f"Textbook setting: shortcut opens at step {textbookConfig.shortcutStep} of {textbookConfig.totalSteps}",
        )
    else:
        print(f"skipping figure 6: no results in {textbookDir} (run: python main.py --textbook)")

    print("saved:")
    for path in savedFiles:
        print(f"  {path}")


if __name__ == "__main__":
    main()
