from .mdp import MDP
from abc import ABC, abstractmethod


class Environment(ABC, MDP):
    """
    Schema of an Environment. An environment is an MDP but has more methods and attributes. It simulates
    the dynamic of an environment.

    For example, a GridWorld env has obstacles, terminal states, non-terminal states, plausible actions
    for a given state s, etc. All of those are modeled and set in its MDP, no need to set manually as if you were
    instantiating directly an MDP
    """
    @abstractmethod
    def iterstates(self):
        pass

    @abstractmethod
    def get_actions(self, state):
        pass
