from .mdp import MDP


class SimpleMDP(MDP):
    """
    This is a simple MDP where for all states, each one can take all actions.
    """
    def __init__(self, states, actions, gamma=0.9):
        super().__init__(states, actions, gamma)

    def iterstates(self) -> 'iterator | list | iterable':
        for state in self.states:
            yield state

    def iteractions(self, state):
        for action in self.actions:
            yield action
