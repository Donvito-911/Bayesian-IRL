import numpy as np


class DP:
    """
    Dynamic programming with algorithms of policy iteration: policy evaluation-policy improvement
    """
    def __init__(self, mdp: 'MDP'):
        self.mdp = mdp
        self.values = np.array([0 for _ in range(self.mdp.n_states)])  # V(s) initialize
        # naive policy
        self.initial_policy = {state: next(self.mdp.iteractions(state)) for state in self.mdp.iterstates()}
        self.rewards = None

    def policy_iteration(self, rewards: np.array, policy=None):  # policy iteration algorithm
        if policy is None:
            policy = self.initial_policy

        self.rewards = rewards
        while True:
            self.policy_evaluation(policy)
            policy_stable = self.policy_improvement(policy)
            if policy_stable:
                break
        return policy

    def policy_evaluation(self, policy, threshold: float = 1e-3):
        while True:
            delta = 0
            for state in self.mdp.iterstates():
                p_state = self.mdp.states[state]
                v = self.values[p_state]
                self.values[p_state] = self.compute_q_s(state, policy[state])
                delta = max(delta, abs(v - self.values[p_state]))
            if delta < threshold:
                break

    def policy_improvement(self, policy):
        policy_stable = True
        for state in self.mdp.iterstates():
            old_action = policy[state]
            policy[state] = self.argmax(state)  
            if old_action != policy[state]:
                policy_stable = False
        return policy_stable

    def argmax(self, state):
        argmax_, maxq = "", -np.inf
        for a in self.mdp.iteractions(state): 
           q = self.compute_q_s(state, a)
           if q > maxq: 
               argmax_, maxq = a, q 

        return argmax_

    def compute_q_s(self, state, action):
        t_probs_s_a = self.mdp.get_transition_prob(state, action)
        sum_ = np.sum(t_probs_s_a * self.values)  # sum of T(s,a, s_) x V(s_) for each s_
        return self.mdp.get_reward(self.rewards, state) + self.mdp.gamma * sum_


    def review_q(self, rewards: np.array, policy: dict) -> bool:
        pass