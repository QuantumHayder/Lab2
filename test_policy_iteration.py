import unittest
from collections import deque

import numpy as np
from gymnasium.utils.env_checker import check_env
from gymnasium.error import ResetNeeded
from gymnasium.wrappers import TimeLimit

from GridWorld import GridWorldEnv
from Policy_Iteration import PolicyIteration


class GridWorldPolicyIterationTests(unittest.TestCase):
    def test_gymnasium_and_reset_contract(self):
        env = GridWorldEnv()
        check_env(env, skip_render_check=True)
        for seed in range(30):
            obs, _ = env.reset(seed=seed)
            self.assertEqual(obs.shape, (8,))
            self.assertTrue(env.observation_space.contains(obs))
            self.assertEqual(len(set(map(tuple, obs.reshape(4, 2)))), 4)
            repeat, _ = env.reset(seed=seed)
            np.testing.assert_array_equal(obs, repeat)
            obs[:] = -1
            self.assertTrue(np.all(env._get_obs() >= 0))

    def test_model_matches_step(self):
        env = GridWorldEnv(3)
        for seed in range(10):
            env.reset(seed=seed)
            self.assertEqual(len(tuple(env.get_states())), 9)
            for state in env.get_states():
                for action in range(4):
                    env._agent_location = np.array(state[:2])
                    before = env._get_obs()
                    predicted, reward, done = env.transition(state, action)
                    np.testing.assert_array_equal(before, env._get_obs())
                    obs, actual_reward, terminated, truncated, _ = env.step(action)
                    self.assertEqual(env.observation_to_state(obs), predicted)
                    self.assertEqual(actual_reward, reward)
                    self.assertEqual(terminated, done)
                    self.assertFalse(truncated)

    def test_rewards_boundaries_and_absorbing_terminals(self):
        env = GridWorldEnv(3)
        layout = dict(agent=[1, 1], target=[1, 2], bad_cells=[[0, 1], [1, 0]])
        for action, expected_reward, outcome in ((0, 1, 'goal'), (1, -1, 'bad_cell'),
                                                  (2, -1, 'bad_cell'), (3, 0, 'ongoing')):
            env.reset(options=layout)
            obs, reward, done, _, info = env.step(action)
            self.assertEqual(reward, expected_reward)
            self.assertEqual(info['outcome'], outcome)
            self.assertEqual(done, outcome != 'ongoing')
            if done:
                next_obs, reward, done, _, _ = env.step(3)
                np.testing.assert_array_equal(obs, next_obs)
                self.assertEqual(reward, 0)
                self.assertTrue(done)
        env.reset(options=dict(agent=[0, 0], target=[2, 2], bad_cells=[[1, 1], [1, 2]]))
        obs, reward, done, _, _ = env.step(1)
        np.testing.assert_array_equal(obs[:2], [0, 0])
        self.assertEqual(reward, 0)
        self.assertFalse(done)

    def test_optimal_values_against_independent_breadth_first_search(self):
        for size in (2, 3, 5):
            for seed in range(30):
                env = GridWorldEnv(size)
                obs, _ = env.reset(seed=seed)
                goal = tuple(obs[2:4])
                bad = {tuple(obs[4:6]), tuple(obs[6:8])}
                distances = {goal: 0}
                queue = deque([goal])
                while queue:
                    row, col = queue.popleft()
                    for dr, dc in ((0, 1), (-1, 0), (0, -1), (1, 0)):
                        nxt = row + dr, col + dc
                        if (0 <= nxt[0] < size and 0 <= nxt[1] < size
                                and nxt not in bad and nxt not in distances):
                            distances[nxt] = distances[(row, col)] + 1
                            queue.append(nxt)
                solver = PolicyIteration(env)
                solver.policy_iteration()
                self.assertTrue(solver.converged)
                for state in solver.states:
                    pos = state[:2]
                    distance = distances.get(pos)
                    expected = solver.gamma ** (distance - 1) if distance else 0.0
                    # Evaluation tolerance bounds value error by roughly theta / (1 - gamma).
                    self.assertAlmostEqual(solver.values[state], expected, delta=1e-7)
                    if distance:
                        current = state
                        for index in range(distance):
                            current, reward, done = env.transition(current, solver.policy[current])
                            self.assertNotIn(current[:2], bad)
                            self.assertEqual(done, index == distance - 1)
                        self.assertEqual(current[:2], goal)
                        self.assertEqual(reward, 1)
                old_policy = solver.policy.copy()
                self.assertEqual(solver.policy_iteration(), 1)
                self.assertEqual(solver.policy, old_policy)

    def test_unreachable_goal_is_not_reported_as_success(self):
        env = GridWorldEnv(3)
        env.reset(options=dict(agent=[0, 0], target=[2, 2], bad_cells=[[0, 1], [1, 0]]))
        solver = PolicyIteration(env)
        solver.policy_iteration()
        state = env.observation_to_state(env._get_obs())
        self.assertAlmostEqual(solver.values[state], 0, delta=1e-7)
        for _ in range(10):
            state, reward, done = env.transition(state, solver.policy[state])
            self.assertEqual(reward, 0)
            self.assertFalse(done)
        wrapped = TimeLimit(env, max_episode_steps=3)
        obs, _ = wrapped.reset(options=dict(
            agent=[0, 0], target=[2, 2], bad_cells=[[0, 1], [1, 0]]))
        for step in range(3):
            obs, reward, terminated, truncated, info = wrapped.step(
                solver.policy[env.observation_to_state(obs)])
            self.assertFalse(terminated)
            self.assertFalse(info['is_success'])
            self.assertEqual(truncated, step == 2)
        wrapped.close()

    def test_rgb_rendering(self):
        env = GridWorldEnv(3, render_mode='rgb_array')
        try:
            env.reset(options=dict(agent=[0, 1], target=[2, 0], bad_cells=[[0, 0], [2, 2]]))
            frame = env.render()
            self.assertEqual(frame.shape, (600, 600, 3))
            self.assertEqual(frame.dtype, np.uint8)
            np.testing.assert_array_equal(frame[100, 300], (45, 105, 220))
            np.testing.assert_array_equal(frame[500, 100], (50, 180, 100))
            np.testing.assert_array_equal(frame[20, 50], (215, 65, 65))
            np.testing.assert_array_equal(frame[420, 450], (215, 65, 65))
            env.step(3)
            np.testing.assert_array_equal(env.render()[300, 300], (45, 105, 220))
        finally:
            env.close()

    def test_discounted_loop_and_validation(self):
        class LoopMDP:
            def get_states(self): return (0,)
            def get_actions(self, state): return (0,)
            def transition(self, state, action): return 0, 1.0, False
        solver = PolicyIteration(LoopMDP(), gamma=0.5)
        self.assertAlmostEqual(solver.policy_evaluation()[0], 2, places=7)
        with self.assertRaises(RuntimeError):
            solver.policy_evaluation(values={0: 0.0}, max_sweeps=1)
        for size in (0, 1, 2.5):
            with self.assertRaises(ValueError): GridWorldEnv(size)
        for gamma in (-1, 1, float('nan')):
            with self.assertRaises(ValueError): PolicyIteration(LoopMDP(), gamma=gamma)
        with self.assertRaises(ValueError): solver.policy_evaluation(theta=0)
        with self.assertRaises(ResetNeeded): PolicyIteration(GridWorldEnv())
        env = GridWorldEnv()
        with self.assertRaises(ValueError):
            env.reset(options=dict(agent=[0, 0], target=[0, 0], bad_cells=[[1, 1], [2, 2]]))


if __name__ == '__main__':
    unittest.main()
