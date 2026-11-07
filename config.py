from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    """
    i keep every setting from the assignment in one place so that both agents are
    always run with exactly the same parameters. i also save this whole object next
    to the results, so anyone reading them can see the exact setup that produced them.
    """

    numRows: int = 6
    numCols: int = 9
    startState: tuple = (5, 3)
    goalState: tuple = (0, 8)
    wallsBefore: tuple = tuple((3, col) for col in range(1, 9))
    wallsAfter: tuple = tuple((3, col) for col in range(1, 8))
    shortcutStep: int = 500
    totalSteps: int = 3000
    gamma: float = 0.95
    alpha: float = 1.0
    epsilon: float = 0.1
    planningSteps: int = 10
    kappa: float = 0.001
    numRuns: int = 30
    baseSeed: int = 0
    resultsDir: str = "results"

    @classmethod
    def fromDict(cls, configDict):
        """
        i rebuild a config from the dictionary saved in summary.json. json turns my
        tuples into lists, so i turn them back into tuples before creating the object.
        """
        cleaned = {}
        for key, value in configDict.items():
            if isinstance(value, list):
                cleaned[key] = tuple(tuple(item) if isinstance(item, list) else item for item in value)
            else:
                cleaned[key] = value
        return cls(**cleaned)

    def toDict(self):
        """
        i turn the config into a plain dictionary with lists instead of tuples, so it
        can be written straight into the json summary.
        """
        configDict = asdict(self)
        for key, value in configDict.items():
            if isinstance(value, tuple):
                configDict[key] = [list(item) if isinstance(item, tuple) else item for item in value]
        return configDict
