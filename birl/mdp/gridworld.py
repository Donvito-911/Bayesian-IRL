from .environment import Environment
import numpy as np


class GridWorld(Environment):
    """
    GridWorld Environment
    """
    def __init__(self, dims: tuple, gamma: float = 0.9, noise: float = 0.2):
        self.board = np.full(dims, ' ', dtype=object)
        self.noise = noise
        self.dimensions = dims
        self.terminals = set()
        self.traps = set()
        actions = ["up", "down", "left", "right", "out"]
        states = [(r, c) for r in range(self.dimensions[0]) for c in range(self.dimensions[1])]
        super().__init__(states, actions, gamma=gamma)  # create the MDP with transition probabilities to 0
        self.__init_transition_probabilities()

    def set_traps(self, traps: list[tuple]) -> None:
        """
        set the traps/obstacles of the gridworld
        :param traps: list of states where each state is a tuple with a coordinate (row, col)
        :return:
        """
        for trap_state in traps:
            self.traps.add(trap_state)

        # reset transitions
        self.t_probabilities = np.zeros(self.t_probabilities.shape)
        self.__init_transition_probabilities()

    def set_terminals(self, terminals: list[tuple]) -> None:
        """
        set the terminal states of the gridworld
        :param terminals: list of states where each state is a tuple with a coordinate (row, col)
        :return:
        """
        for terminal_state in terminals:
            self.terminals.add(terminal_state)
            # update transitions of terminal state
            pointer_terminal_state, pointer_out = self.states[terminal_state], self.actions["out"]
            self.t_probabilities[pointer_terminal_state, :, :] = 0
            self.t_probabilities[pointer_terminal_state, pointer_out, pointer_terminal_state] = 1

    def iterstates(self, include_terminals: bool = False):
        """
        iterate over non-terminal and non-blocking states
        """
        for r in range(self.dimensions[0]):
            for c in range(self.dimensions[1]):
                if (r, c) not in self.traps and ((r, c) not in self.terminals or include_terminals):
                    yield r, c

    def iteractions(self, state: tuple):
        """
        iterate the possible actions that can be taken in a state (ignore the actions that are not plausible, such as
        going out of bounds or going to an obstacle)
        :param state: tuple in the form (row, col)
        :return:
        """
        if state in self.traps:
            return

        if state in self.terminals:
            yield "out"
            return

        row, col = state
        if row - 1 >= 0 and (row-1, col) not in self.traps:
            yield "up"
        if row + 1 < self.dimensions[0] and (row + 1, col) not in self.traps:
            yield "down"
        if col - 1 >= 0 and (row, col - 1) not in self.traps:
            yield "left"
        if col + 1 < self.dimensions[1] and (row, col + 1) not in self.traps:
            yield "right"

    def __init_transition_probabilities(self):
        """
        Initialize the transition probabilities of the MDP given the dynamics of the environment.
        :return:
        """
        for state in self.iterstates():
            for action in self.iteractions(state):
                transition_probs = self.__generate_transition_prob(state, action)
                self.set_transition_probabilities(transition_probs)

    def __generate_transition_prob(self, s: tuple, a: str) -> list[tuple['state', 'action', 'state', float]]:
        """
        generate the transition probabilities for a given state 's' and an action 'a' taken
        :param s: state in the form of tuple (row, col)
        :param a: action as a string
        :return: a list of transition probabilities for a specific state-action
        """
        noise_actions = {"up": ["left", "right"],
                         "right": ["up", "down"],
                         "left": ["up", "down"],
                         "down": ["left", "right"]}
        transition_prob = {(s, a, self.__get_next_state(s, a)): 1 - self.noise}
        for n_a in noise_actions[a]:
            s_ = self.__get_next_state(s, n_a)
            transition_prob[(s, a, s_)] = transition_prob.get((s, a, s_), 0) + self.noise/len(noise_actions[a])
        return [(s, a, s_, prob) for (s, a, s_), prob in transition_prob.items()]

    def __get_next_state(self, state: tuple, action: str) -> tuple:
        """
        get the new state when an action is executed in a given state (without noise). If the state is not
        reachable, the new state is actually the old state.
        :param state: the initial state
        :param action: the action taken in the initial state
        :return: tuple that represents the new state
        """
        r, c = state
        new_states = {"up": (r-1, c), "down": (r+1, c), "left": (r, c-1), "right": (r, c+1)}
        new_state = new_states[action]
        if self.__is_reachable_state(new_state):
            return new_state
        return state

    def __is_reachable_state(self, state: tuple) -> bool:
        """
        verify the state is reachable (i.e. the state is actually in the board of gridworld and it is not
        an obstacle/trap)
        :param state: state to be verified
        :return: True if state is reachable, False otherwise
        """
        if state in self.traps or state not in self.states:
            return False
        return True

    def show(self) -> None:
        """
        display the gridworld in a user-friendly way
        :return:
        """
        copy = self.board.copy()
        for r in range(self.dimensions[0]):
            for c in range(self.dimensions[1]):
                if (state := (r, c)) in self.traps:
                    copy[state] = "*"
                elif (state := (r, c)) in self.terminals:
                    copy[state] = "T"

        labels = np.arange(0, self.dimensions[1])
        print('  %s ' % (' '.join('%04s' % i for i in labels)))
        print('  .%s.' % ('-'.join('%04s' % "---" for i in labels)))
        for row_label, row in zip(labels, copy):
            print ('%s |%s|' % (row_label, '|'.join('%04s' % i for i in row)))
            print ('  |%s|' % ('|'.join('%01s' % "----" for i in row)))

