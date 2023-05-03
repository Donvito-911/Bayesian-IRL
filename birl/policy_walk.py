from utils.auxiliar_func import get_random_reward_vector
from utils.dynamic_programming import DP
import numpy as np


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
        update = dp.compute_q(policy, r_2)  # TODO: implementar
        # c)
        if update:
            policy_2 = dp.policy_iteration(r, policy)
            prob = min(1, distribution.p(r_2, policy_2)/distribution.p(r, policy))
            if np.random.uniform() <= prob:
                r, policy = r_2, policy_2
        else:
            prob = min(1, distribution.p(r_2, policy) / distribution.p(r, policy))
            if np.random.uniform() <= prob:
                r = r_2

    return r
