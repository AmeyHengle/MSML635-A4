import numpy as np


class DynaQAgent:
    """
    i implement tabular dyna-q. every real step does three things: a q-learning update
    from the real transition (direct rl), an update of a deterministic model that
    remembers the last outcome of each state-action pair, and a number of planning
    updates that replay randomly chosen transitions from that model.
    """

    displayName = "Dyna-Q"

    def __init__(self, numRows, numCols, numActions, alpha, gamma, epsilon, planningSteps, rng):
        """
        i start every q-value at zero and the model empty. the model maps a
        (row, col, action) key to the reward, next state and whether that next state
        was the goal. i also keep a list of the keys so i can sample one uniformly
        during planning without converting the dictionary every time.
        """
        self.numActions = numActions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planningSteps = planningSteps
        self.rng = rng
        self.qValues = np.zeros((numRows, numCols, numActions))
        self.model = {}
        self.modelKeys = []

    def greedyAction(self, state):
        """
        i return the action with the highest q-value in this state. when several
        actions tie, which happens a lot at the start because everything is zero, i
        pick one of them at random so the agent does not keep choosing the first action.
        """
        actionValues = self.qValues[state[0], state[1]]
        bestActions = np.flatnonzero(actionValues == actionValues.max())
        return int(self.rng.choice(bestActions))

    def selectAction(self, state):
        """
        i use epsilon-greedy selection: with probability epsilon i take a uniformly
        random action, otherwise i take the greedy action.
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.numActions))
        return self.greedyAction(state)

    def observeState(self, state, timeStep):
        """
        i call this right before the agent acts in a state. plain dyna-q has nothing to
        do here, but dyna-q+ uses it to add placeholder model entries the first time it
        sees a state.
        """
        return None

    def qLearningUpdate(self, state, action, reward, nextState, isTerminal):
        """
        i apply the one-step q-learning update. if the next state is the goal i do not
        bootstrap, because the episode ends there and the goal has no future value, so
        the target is just the reward.
        """
        target = reward
        if not isTerminal:
            target += self.gamma * self.qValues[nextState[0], nextState[1]].max()
        currentValue = self.qValues[state[0], state[1], action]
        self.qValues[state[0], state[1], action] = currentValue + self.alpha * (target - currentValue)

    def updateModel(self, state, action, reward, nextState, isTerminal, timeStep):
        """
        i store the latest real outcome of this state-action pair. the maze is
        deterministic so remembering the last outcome is enough, and it also means a
        new outcome simply overwrites an old or placeholder one.
        """
        modelKey = (state[0], state[1], action)
        if modelKey not in self.model:
            self.modelKeys.append(modelKey)
        self.model[modelKey] = (reward, nextState, isTerminal)

    def planningReward(self, modelKey, reward, timeStep):
        """
        i return the reward that planning should use for a model transition. for
        dyna-q this is just the reward stored in the model.
        """
        return reward

    def plan(self, timeStep):
        """
        i run the planning loop: i pick a previously seen state-action pair uniformly
        at random, ask the model what happened, and apply the same q-learning update
        as if it were real experience. i repeat this n times per real step, which is
        10 in this assignment.
        """
        if not self.modelKeys:
            return
        for _ in range(self.planningSteps):
            modelKey = self.modelKeys[self.rng.integers(len(self.modelKeys))]
            reward, nextState, isTerminal = self.model[modelKey]
            simulatedReward = self.planningReward(modelKey, reward, timeStep)
            simulatedState = (modelKey[0], modelKey[1])
            self.qLearningUpdate(simulatedState, modelKey[2], simulatedReward, nextState, isTerminal)

    def learn(self, state, action, reward, nextState, isTerminal, timeStep):
        """
        i run one full dyna step after a real transition: direct rl first, then the
        model update, then planning. the direct update always uses the real reward,
        any exploration bonus only ever shows up inside planning.
        """
        self.qLearningUpdate(state, action, reward, nextState, isTerminal)
        self.updateModel(state, action, reward, nextState, isTerminal, timeStep)
        self.plan(timeStep)


class DynaQPlusAgent(DynaQAgent):
    """
    i implement dyna-q+ on top of dyna-q. the only changes are that i remember when
    each state-action pair was last executed for real, i add kappa * sqrt(tau) to the
    reward during planning, and i give untried actions placeholder model entries so
    planning can push the agent to go and try them.
    """

    displayName = "Dyna-Q+"

    def __init__(self, numRows, numCols, numActions, alpha, gamma, epsilon, planningSteps, rng, kappa):
        """
        i set everything up like dyna-q and add kappa, a table of the time step each
        pair was last tried and a set of states i have already visited. the table
        starts at zero for every pair, so a pair that has never been tried counts its
        tau from the beginning of the run, as the assignment asks.
        """
        super().__init__(numRows, numCols, numActions, alpha, gamma, epsilon, planningSteps, rng)
        self.kappa = kappa
        self.lastTried = np.zeros((numRows, numCols, numActions))
        self.visitedStates = set()

    def observeState(self, state, timeStep):
        """
        the first time i visit a state, i add all four of its actions to the model as
        if they had been tried once, led back to the same state and gave a reward of 0
        (sutton and barto, section 8.3). i only do this for states the agent actually
        acts in, so the goal never gets placeholder entries. when an action is really
        taken, the model update overwrites its placeholder with the true outcome.
        """
        if state in self.visitedStates:
            return
        self.visitedStates.add(state)
        for action in range(self.numActions):
            modelKey = (state[0], state[1], action)
            if modelKey not in self.model:
                self.modelKeys.append(modelKey)
                self.model[modelKey] = (0.0, state, False)

    def updateModel(self, state, action, reward, nextState, isTerminal, timeStep):
        """
        i store the real outcome like dyna-q does and record the current time step as
        the last time this pair was executed, which resets its tau to zero.
        """
        super().updateModel(state, action, reward, nextState, isTerminal, timeStep)
        self.lastTried[state[0], state[1], action] = timeStep

    def planningReward(self, modelKey, reward, timeStep):
        """
        i add the exploration bonus kappa * sqrt(tau), where tau is the number of real
        time steps since this pair was last executed. pairs the agent has ignored for a
        long time get a bigger bonus, which makes them look worth revisiting.
        """
        tau = timeStep - self.lastTried[modelKey]
        return reward + self.kappa * np.sqrt(tau)
