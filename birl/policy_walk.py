from utils.auxiliar_func import get_random_reward_vector
from utils.dynamic_programming import DP
import numpy as np


class PolicyWalk:
    def __init__(self, distribution: 'Distribution', mdp: 'Environment',
                 delta: float = 0.1, n_iter: int = 1_000):
        self.distribution = distribution
        self.mdp = mdp
        self.delta = delta
        self.n_iter = n_iter

    def execute(self) -> list:
        """

        :param distribution: Prior distribution to use
        :param mdp: the Markov Decision Process
        :param delta: step size delta for generating the sample of the space reward functions
        :param n_iter: total of iterations
        :return:
        """
        # step 1: Pick a random reward vector r
        r = get_random_reward_vector(self.mdp.n_states, delta=self.delta)
        # step 2: policy iteration
        dp = DP(self.mdp)
        policy = dp.policy_iteration(r)
        # step 3:
        for _ in range(n_iter):
            # a) reward vector from neighbours
            r_2 = get_random_reward_vector(r, delta=delta)
            # b) compute Q
            is_updatable = dp.review_q(r_2, policy)  # TODO: implementar
            # c)
            if is_updatable:
                policy_2 = dp.policy_iteration(r, policy)
                prob = min(1, distribution.p(r_2, policy_2) / distribution.p(r, policy))
                if np.random.uniform() <= prob:
                    r, policy = r_2, policy_2
            else:
                prob = min(1, distribution.p(r_2, policy) / distribution.p(r, policy))
                if np.random.uniform() <= prob:
                    r = r_2

        return r

    def __generate_random_reward_vector(self, type_: int | np.array, r_min: float = 0, r_max: float = 1) -> np.array:
        """
        generate a new random reward vector (with
        :param type_:
        :return:
        """
        if isinstance(n_states := type_, int):  # pick a random reward vector R
            possible_rewards = np.arange(r_min, r_max + self.delta, self.delta)
            return np.array([np.random.choice(possible_rewards) for _ in n_states])

        # pick a random reward vector R_2 from the neighbours of R (random_vector)
        random_vector = type_
        f_neighbors = lambda x: x + np.random.choice([0, self.delta, -self.delta])
        return np.array(list(map(f_neighbors, random_vector)))


def policy_walk(distribution: 'Distribution', mdp: 'Environment', delta: float = 0.1, n_iter: int = 1_000) -> list:
    """

    :param distribution: Prior distribution to use
    :param mdp: the Markov Decision Process
    :param delta: step size delta for generating the sample of the space reward functions
    :param n_iter: total of iterations
    :return:
    """
    # step 1: Pick a random reward vector r
    r = get_random_reward_vector(mdp.n_states, delta=delta)
    # step 2: policy iteration
    dp = DP(mdp)
    dp.policy_iteration(r)
    policy = dp.policy_iteration(mdp, r)
    # step 3:
    for _ in range(n_iter):
        # a) reward vector from neighbours
        r_2 = get_random_reward_vector(r, delta=delta)
        # b) compute Q
        is_updatable = dp.review_q(r_2, policy)  # TODO: implementar
        # c)
        if is_updatable:
            policy_2 = dp.policy_iteration(r, policy)
            prob = min(1, distribution.p(r_2, policy_2)/distribution.p(r, policy))
            if np.random.uniform() <= prob:
                r, policy = r_2, policy_2
        else:
            prob = min(1, distribution.p(r_2, policy) / distribution.p(r, policy))
            if np.random.uniform() <= prob:
                r = r_2

    return r
