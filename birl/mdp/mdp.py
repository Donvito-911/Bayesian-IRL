import numpy as np


class MDP:
    """
    The Markov Decision Process as a 4-tuple (S, A, T, gamma) where
    S is a set of finite states.
    A is a set of finite actions.
    T is a transition probability function (matrix) of the form T: S x A x S -> [0,1]
    gamma is the discount factor in the range  [0,1).

    This is an API for manipulating the MDP of an Environment (such as GridWorld). But it can be instantiated to see
    the dynamics of any MDP as previously described. Nevertheless, this MDP is not sufficient to work with the BIRL
    algorithm policy_walk (it uses an Environment, which has more methods than an MDP).
    """
    def __init__(self, states: list, actions: list, gamma: float = 0.9):
        """
        Constructor of the MDP
        :param states: list of states (S)
        :param actions: list of actions (A)
        :param gamma: discount factor
        """
        # states and actions are saved with dict structure because it saves a pointer useful for transition prob. matrix
        self.states = {state: pointer for pointer, state in enumerate(states)}
        self.actions = {action: pointer for pointer, action in enumerate(actions)}
        self.gamma = gamma
        self.n_states = len(states)
        self.n_actions = len(actions)
        # the probability transitions are initialized in 0
        self.t_probabilities = np.zeros((self.n_states, self.n_actions, self.n_states))

    def get_transition_prob(self, s, a=None, s_=None) -> 'float | np.array':
        """
        get the transition probability of reaching a state 's_' when the action 'a' was taken in state 's'
        If s_ is none then it returns an array with all the transition probabilities states
        for a state 's' and an action 'a'.
        If a_ is none then it returns an array with all the transition probabilities for each action from state s to
        state s_
        If a_ and s_ are None, it returns a matrix transition probabilities for each action and each state of a given
        state 's'.
        :param s: initial state
        :param a: action taken in initial state
        :param s_: destiny state
        :return: probability in range [0,1] or array with transition probabilities
        """
        self.__is_valid_sas(s, a, s_)
        if a is None and s_ is None:
            s = self.states[s]  # get the pointer/index in transition matrix
            return self.t_probabilities[s, :, :]
        elif s_ is None:
            s, a = self.states[s], self.actions[a]
            return self.t_probabilities[s, a, :]
        elif a is None:
            s, s_ = self.states[s], self.states[s_]
            return self.t_probabilities[s, :, s_]

        s, a, s_ = self.states[s], self.actions[a], self.states[s_]
        return self.t_probabilities[s, a, s_]

    def set_transition_probability(self, prob: float, s: 'state', a: 'action' = None, s_: 'state' = None):
        """
        Set the transition probability for a state 's' with an action 'a' and reaching state 's_'
        :param s: initial state
        :param a: action taken in initial state
        :param s_: destiny state
        :param prob: probability of t(s, a, s_) in range [0,1]
        :return: None
        """
        self.__is_valid_sas(s, a, s_)
        if a is None and s_ is None:
            s = self.states[s]  # get the pointer/index in transition matrix
            self.t_probabilities[s, :, :] = prob
        elif s_ is None:
            s, a = self.states[s], self.actions[a]
            self.t_probabilities[s, a, :] = prob
        elif a is None:
            s, s_ = self.states[s], self.states[s_]
            self.t_probabilities[s, :, s_] = prob
        else:
            s, a, s_ = self.states[s], self.actions[a], self.states[s_]
            self.t_probabilities[s, a, s_] = prob

    def __is_valid_sas(self, s, a, s_):
        if s not in self.states:
            raise Exception(f'The state {s} was not found in states: {set(self.states.keys())}')
        if a is not None and a not in self.actions:
            raise Exception(f'The action {a} was not found in actions: {set(self.actions.keys())}')
        if s_ is not None and s_ not in self.states:
            raise Exception(f'The state {s_} was not found in states: {set(self.states.keys())}')

    def set_transition_probabilities(self, lst: list[tuple['state', 'action', 'state', float]]) -> None:
        """
        set the transition probabilities of a given list of transition probabilities of the form
        [(s, a, s_, prob), ...]

        :param lst: list with transition probabilities
        :return: None
        """

        for s, a, s_, prob in lst:
            self.set_transition_probability(prob, s, a, s_)

    def generate_random_transition_probabilities(self) -> None:
        """
        Generate random transition probability for all the state-action-state_
        :return: None
        """
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.t_probabilities[s, a] = self.__generate_random_vector(self.n_states)

    def reset_transitions(self):
        """
        reset the transition probabilities to 0.
        :return:
        """
        self.t_probabilities = np.zeros(self.t_probabilities.shape)

    def get_reward(self, rewards: np.array, s):
        """
        Given an external array of rewards R of size |S|, get the reward of specific state
        :param rewards: array of floats R
        :param s: state
        :return: R(s)
        """
        return rewards[self.states[s]]

    @staticmethod
    def __generate_random_vector(size: int):
        """
        get a random vector that sums up to 1
        :param size: size of the vector
        :return: a vector of size 'size' when sum of the vector equals 1 and each element >=0
        """
        return np.random.dirichlet(np.ones(size))
