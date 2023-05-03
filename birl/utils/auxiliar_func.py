import numpy as np


def get_random_reward_vector(variable: int | np.array, r_min: float = -1, r_max: float = 1,
                             delta: float = 0.1) -> np.array:
    if isinstance(n_states := variable, int):  # pick a random reward vector R
        possible_rewards = np.arange(r_min, r_max + delta, delta)
        return np.array([np.random.choice(possible_rewards) for _ in n_states])

    # pick a random reward vector R_2 from the neighbours of R (random_vector)
    random_vector = variable
    f_neighbors = lambda x: x + np.random.choice([0, delta, -delta])
    return np.array(list(map(f_neighbors, random_vector)))
