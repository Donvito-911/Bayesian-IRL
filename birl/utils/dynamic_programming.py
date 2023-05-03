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
            policy[state] = None  # TODO implementar argmax
            if old_action != policy[state]:
                policy_stable = False
        return policy_stable

    def compute_v_s(self, state, policy):
        # transition probabilities of state-action (the size are states)
        t_probs_s_a = self.mdp.get_transition_prob(state, policy[state])
        sum_ = np.sum(t_probs_s_a * self.values)  # sum of T(s,a, s_) x V(s_) for each s_
        return self.mdp.get_reward(self.rewards, state) + self.mdp.gamma * sum_
