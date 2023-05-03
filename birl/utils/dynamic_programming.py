import numpy as np

class DP:
    """
    Dynamic programming with algorithms of policy iteration: policy evaluation-policy improvement
    """
    def __init__(self, mdp: 'Environment'):
        self.mdp = mdp
        # self.values = {state: 0 for state in self.mdp.iterstates(include_terminals=True)}
        self.values = np.array([0 for _ in range(self.mdp.n_states)])
        self.initial_policy = {state: next(self.mdp.iteractions()) for state in self.mdp.iterstates()}
        self.rewards = None

    def policy_iteration(self, rewards: np.array, policy=None):
        if policy is None:
            policy = self.initial_policy

        self.rewards = rewards
        while True:
            self.policy_evaluation(policy)
            policy_stable = self.policy_improvement(policy)
            if policy_stable:
                break
        return policy

    def policy_evaluation(self, policy, threshold: float = 1e-4):
        while True:
            delta = 0
            for state in self.mdp.iterstates():
                v = self.values[state]
                self.values[state] = self.compute_v_s(state, policy)
                delta = max(delta, abs(v - self.values[state]))
            if delta < threshold:
                break

    def policy_improvement(self, policy):
        policy_stable = True
        for state in self.mdp.iterstates():
            old_action = policy[state]
            policy[state] = None  # TODO implementar
            if old_action != policy[state]:
                policy_stable = False
        return policy_stable

    def compute_v_s(self, state, policy):
        p_state, p_policy = self.mdp.states[state], self.mdp.actions[policy[state]]
        sum_ = np.sum(self.mdp.t_probabilities[p_state, p_policy, :] * self.values)
        return self.rewards[p_state] + self.mdp.gamma * sum_
