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

    def get_transition_probability(self, s, a, s_) -> float:
        """
        get the transition probability of reaching a state 's_' when the action 'a' was taken in state 's'
        :param s: initial state
        :param a: action taken in initial state
        :param s_: destiny state
        :return: probability in range [0,1]
        """
        s, a, s_ = self.__get_t_pos(s, a, s_)  # get the pointers of s, a, and s_
        return self.t_probabilities[s, a, s_]

    def set_transition_probability(self, s: 'state', a: 'action', s_: 'state', prob: float):
        """
        Set the transition probability for a state 's' with an action 'a' and reaching state 's_'
        :param s: initial state
        :param a: action taken in initial state
        :param s_: destiny state
        :param prob: probability of t(s, a, s_) in range [0,1]
        :return: None
        """
        s, a, s_ = self.__get_t_pos(s, a, s_)  # get the pointers of s, a, and s_
        self.t_probabilities[s, a, s_] = prob

    def set_transition_probabilities(self, lst: list[tuple['state', 'action', 'state', float]]) -> None:
        """
        set the transition probabilities of a given list of transition probabilities of the form
        [(s, a, s_, prob), ...]

        :param lst: list with transition probabilities
        :return: None
        """
        for s, a, s_, prob in lst:
            self.set_transition_probability(s, a, s_, prob)

    def generate_random_transition_probabilities(self) -> None:
        """
        Generate random transition probability for all the state-action-state_
        :return: None
        """
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.t_probabilities[s, a] = self.__generate_random_vector(self.n_states)

    def __get_t_pos(self, s, a, s_) -> list['pointer_state', 'pointer_action', 'pointer_state']:
        """
        Get the pointers of states s, s_ and action, so it can be referenced in probability transition matrix
        :param s: state s
        :param a: action taken in state s
        :param s_: state s_
        :return: pointers referenced in transition matrix of s, a, s_

        Example:
        if states are ['A', 'B', 'C'] and actions are ['a1', 'a2']. Then calling __get_t_pos('A', 'a1', 'B') will return
         [0, 0, 1] because 'A' => 0, 'B' => 1, 'C' => 2 and 'a1' => 0, 'a2' => 1.
        This is useful because the transition probabilities is a matrix, so you need the respective mapping to get
        the probability transition.

        Under the hood, calling the get_transition_probability('A', 'a1', 'B') just
        returns the value at the matrix of the transition probabilities in position [0, 0, 1].
        """
        t_positions = []
        for name in [s, a, s_]:
            if name in self.states:
                t_positions.append(self.states[name])
            elif name in self.actions:
                t_positions.append(self.actions[name])
            else:
                raise Exception(f'The name {name} was not found in states {self.states.keys()} or actions '
                                f'{self.actions.keys()}')

        return t_positions

    @staticmethod
    def __generate_random_vector(size: int):
        """
        get a random vector that sums up to 1
        :param size: size of the vector
        :return: a vector of size 'size' when sum of the vector equals 1 and each element >=0
        """
        return np.random.dirichlet(np.ones(size))
