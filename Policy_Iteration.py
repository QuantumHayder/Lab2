"""Tabular dynamic programming for a deterministic, finite GridWorld."""

import math


class PolicyIteration:
    """Evaluate and improve a deterministic policy using the transition model.

    Policies may be dictionaries or objects with select_action(state, actions)
    and update(state, action). Values are exposed as a state-to-value dictionary.
    Discounting ensures evaluation converges even for policies that loop.
    """

    def __init__(self, mdp, policy=None, gamma=0.9):
        if not math.isfinite(gamma) or not 0 <= gamma < 1:
            raise ValueError("gamma must be finite and in [0, 1)")
        self.mdp = mdp
        self.gamma = gamma
        self.states = tuple(mdp.get_states())
        # Cache the fixed layout's deterministic model once for every backup.
        self.actions = {state: tuple(mdp.get_actions(state)) for state in self.states}
        self.model = {(state, action): mdp.transition(state, action)
                      for state in self.states for action in self.actions[state]}
        self.policy = {} if policy is None else policy
        self.values = dict.fromkeys(self.states, 0.0)
        self.converged = False
        if isinstance(self.policy, dict):
            for state in self.states:
                actions = self.actions[state]
                if actions:
                    self.policy.setdefault(state, actions[0])

    def _action(self, policy, state, actions):
        action = (policy[state] if isinstance(policy, dict)
                  else policy.select_action(state, actions))
        if action not in actions:
            raise ValueError(f"Invalid policy action {action!r} for state {state}")
        return action

    @staticmethod
    def _value(values, state):
        return values[state] if isinstance(values, dict) else values.get_value(state)

    @staticmethod
    def _set_value(values, state, value):
        if isinstance(values, dict):
            values[state] = value
        else:
            values.add(state, value)

    def _q_value(self, values, state, action):
        next_state, reward, terminated = self.model[state, action]
        return reward + (0.0 if terminated else self.gamma * self._value(values, next_state))

    def policy_evaluation(self, policy=None, values=None, theta=1e-8,
                          max_sweeps=10000):
        """Apply in-place Bellman expectation backups until changes are < theta."""
        if not math.isfinite(theta) or theta <= 0:
            raise ValueError("theta must be finite and positive")
        if not isinstance(max_sweeps, int) or max_sweeps < 1:
            raise ValueError("max_sweeps must be a positive integer")
        policy = self.policy if policy is None else policy
        values = self.values if values is None else values
        for _ in range(max_sweeps):
            delta = 0.0
            for state in self.states:
                actions = self.actions[state]
                old_value = self._value(values, state)
                new_value = (self._q_value(values, state, self._action(policy, state, actions))
                             if actions else 0.0)
                self._set_value(values, state, new_value)
                delta = max(delta, abs(old_value - new_value))
            if delta < theta:
                return values
        raise RuntimeError("Policy evaluation did not converge within max_sweeps")

    def policy_iteration(self, max_iterations=100, theta=1e-8):
        """Update the policy in place; return the number of improvement rounds.

        self.converged distinguishes stability from hitting the iteration limit.
        self.values always evaluates the final returned policy.
        """
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.converged = False
        for iteration in range(1, max_iterations + 1):
            self.policy_evaluation(theta=theta)
            policy_changed = False
            for state in self.states:
                actions = self.actions[state]
                if not actions:
                    continue
                old_action = self._action(self.policy, state, actions)
                q_values = {a: self._q_value(self.values, state, a) for a in actions}
                best_action = max(actions, key=q_values.get)
                # Preserve the current action on ties to avoid cycling among optima.
                new_action = (old_action if math.isclose(
                    q_values[old_action], q_values[best_action], rel_tol=0.0, abs_tol=1e-12
                ) else best_action)
                if new_action != old_action:
                    if isinstance(self.policy, dict):
                        self.policy[state] = new_action
                    else:
                        self.policy.update(state, new_action)
                    policy_changed = True
            if not policy_changed:
                self.converged = True
                return iteration
        self.policy_evaluation(theta=theta)
        return max_iterations
