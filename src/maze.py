import heapq
from collections import deque


class ShortcutMaze:
    """
    represent the 6x9 maze as a grid of (row, col) cells with row 0 at the top.
    the maze starts with a wall across row 3 that only leaves column 0 open, and i
    switch to a shorter wall when the shortcut opens. the only thing that changes
    from step to step is where the agent is.
    """

    actionMoves = ((-1, 0), (1, 0), (0, -1), (0, 1))
    actionNames = ("up", "down", "left", "right")

    def __init__(self, numRows, numCols, startState, goalState, wallsBefore, wallsAfter):
        """
        store the grid size, the start and goal cells and both wall layouts. the
        shortcut cells are whatever is a wall before the change but not after it,
        which for this assignment is just (3, 8). begin with the original layout
        and the agent on the start cell.
        """
        self.numRows = numRows
        self.numCols = numCols
        self.startState = tuple(startState)
        self.goalState = tuple(goalState)
        self.wallsBefore = frozenset(tuple(cell) for cell in wallsBefore)
        self.wallsAfter = frozenset(tuple(cell) for cell in wallsAfter)
        self.shortcutCells = self.wallsBefore - self.wallsAfter
        self.numActions = len(self.actionMoves)
        self.walls = self.wallsBefore
        self.shortcutOpen = False
        self.agentState = self.startState

    def reset(self):
        """
        put the agent back on the start cell and return it. deliberately leave the
        walls alone here, so going back to the start after reaching the goal keeps
        whatever layout the maze currently has.
        """
        self.agentState = self.startState
        return self.agentState

    def openShortcut(self):
        """
        switch to the second wall layout, which removes the wall at (3, 8). the agent
        is never told about this, so the only way it can learn about the shortcut is by
        actually moving into that cell.
        """
        self.walls = self.wallsAfter
        self.shortcutOpen = True

    def nextCell(self, cell, action, walls=None):
        """
        work out where an action leads from a cell under a given wall layout. if the
        target cell is off the grid or is a wall, return the same cell, so bumping
        into something just wastes the time step.
        """
        if walls is None:
            walls = self.walls
        rowMove, colMove = self.actionMoves[action]
        newRow = cell[0] + rowMove
        newCol = cell[1] + colMove
        insideGrid = 0 <= newRow < self.numRows and 0 <= newCol < self.numCols
        if not insideGrid or (newRow, newCol) in walls:
            return cell
        return (newRow, newCol)

    def isShortcutCrossing(self, state, nextState):
        """
        count a move as using the shortcut only when the agent steps into a shortcut
        cell from the start's side of the wall. this way an agent near the goal that
        randomly steps down into (3, 8) from above is not counted as having found the
        shortcut, because it has not actually used it as a route from the start.
        """
        if nextState not in self.shortcutCells:
            return False
        wallRow = nextState[0]
        return (state[0] - wallRow) * (self.startState[0] - wallRow) > 0

    def step(self, action):
        """
        move the agent one cell in the chosen direction, where walls and edges keep it
        in place, and give a reward of 1 only when it lands on the goal, otherwise 0. also return whether the goal was reached so the experiment
        loop knows to send the agent back to the start.
        """
        nextState = self.nextCell(self.agentState, action)
        reachedGoal = nextState == self.goalState
        reward = 1.0 if reachedGoal else 0.0
        self.agentState = nextState
        return nextState, reward, reachedGoal

    def shortestPathLength(self, walls=None):
        """
        run a breadth-first search from the start to the goal to find the length of
        the best possible route. use it to check that my maze is built correctly
        (16 steps before the shortcut and 10 after) and to know the best reward rate an
        agent could reach in each phase.
        """
        if walls is None:
            walls = self.walls
        distances = {self.startState: 0}
        frontier = deque([self.startState])
        while frontier:
            cell = frontier.popleft()
            if cell == self.goalState:
                return distances[cell]
            for action in range(self.numActions):
                neighbour = self.nextCell(cell, action, walls)
                if neighbour not in distances:
                    distances[neighbour] = distances[cell] + 1
                    frontier.append(neighbour)
        return None

    def shortestPath(self, walls=None):
        """
        find one shortest route from the start to the goal as a list of cells, for
        drawing in the report. many routes have the same length, so among those pick
        the one with the fewest turns, which is the easiest to read on a figure. do
        this with a small dijkstra search where the cost is the number of steps first
        and the number of direction changes second.
        """
        if walls is None:
            walls = self.walls
        startKey = (self.startState, None)
        bestCost = {startKey: (0, 0)}
        parents = {startKey: None}
        frontier = [(0, 0, self.startState, -1)]
        while frontier:
            steps, turns, cell, lastAction = heapq.heappop(frontier)
            stateKey = (cell, None if lastAction < 0 else lastAction)
            if bestCost.get(stateKey) != (steps, turns):
                continue
            if cell == self.goalState:
                path = []
                while stateKey is not None:
                    path.append(stateKey[0])
                    stateKey = parents[stateKey]
                return path[::-1]
            for action in range(self.numActions):
                neighbour = self.nextCell(cell, action, walls)
                if neighbour == cell:
                    continue
                newCost = (steps + 1, turns + int(lastAction >= 0 and action != lastAction))
                neighbourKey = (neighbour, action)
                if neighbourKey not in bestCost or newCost < bestCost[neighbourKey]:
                    bestCost[neighbourKey] = newCost
                    parents[neighbourKey] = stateKey
                    heapq.heappush(frontier, (newCost[0], newCost[1], neighbour, action))
        return None

    def render(self, walls=None):
        """
        draw the maze as text, marking the start and goal with the same letters the
        assignment uses, walls with # and open cells with a dot. save this next to
        the results so the layouts actually used are on record.
        """
        if walls is None:
            walls = self.walls
        lines = ["   " + " ".join(str(col) for col in range(self.numCols))]
        for row in range(self.numRows):
            rowSymbols = []
            for col in range(self.numCols):
                cell = (row, col)
                if cell == self.startState:
                    rowSymbols.append("S")
                elif cell == self.goalState:
                    rowSymbols.append("G")
                elif cell in walls:
                    rowSymbols.append("#")
                else:
                    rowSymbols.append(".")
            lines.append(f"{row}  " + " ".join(rowSymbols))
        return "\n".join(lines)
