# Assignment 2: Grid Maze policy iteration

A Gymnasium environment with a random agent, goal, and two stationary bad cells. PyGame renders the grid; Gymnasium RecordVideo saves policy rollouts.

## Setup and run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --show
```

Omit `--show` to record without a display. Options: `--episodes 3 --seed 42 --size 5 --max-steps 20`. Each episode selects one maze using seed + episode index and records two videos with identical starting positions: `before-learning` uses the solver's initial all-right policy; `after-learning` uses its converged policy. Each video is in its own labeled subdirectory under `videos/<timestamp>/`. The accompanying `results.json` contains paired trajectories, actions, rewards, and outcomes. Without `--max-steps`, each rollout is limited to `4 * size * size` steps.

The initial policy may hit a bad cell, loop at a boundary, or occasionally succeed. It is not forced to fail. These before/after rollouts visualize the policies; dynamic programming still learns from the transition model, not from rollout failures.

Blue circle = agent, green cell = goal, red X cells = bad cells. Eight-integer observations are `[agent_row, agent_col, goal_row, goal_col, bad1_row, bad1_col, bad2_row, bad2_col]`. All four initial positions are distinct.

Reaching the goal gives +1 and terminates successfully. Entering either bad cell gives -1 and terminates in failure. Other moves give 0, and boundary moves stay in place. Terminal states have no further reward. Bad-cell termination is an explicit assumption because the PDF only specifies its reward. Random layouts can be unsolvable; the rollout wrapper truncates after `4 * size * size` steps and reports `time_limit` separately from success.

## Using the solver

```python
from GridWorld import GridWorldEnv
from Policy_Iteration import PolicyIteration

env = GridWorldEnv(size=5)
observation, info = env.reset(seed=42)
solver = PolicyIteration(env, gamma=0.9)
rounds = solver.policy_iteration()
action = solver.policy[env.observation_to_state(observation)]
observation, reward, terminated, truncated, info = env.step(action)
env.close()
```

Reset before constructing the solver. It plans over all agent positions for that fixed goal/bad-cell layout and caches its transition model. Construct a new solver after a reset changes the layout. Policy iteration uses exact model transitions, not exploration or sampled training.

## Tests and deliverables

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_policy_iteration
```

- `GridWorld.py`: environment, deterministic model, PyGame renderer.
- `Policy_Iteration.py`: policy evaluation and improvement.
- `main.py`: seeded experiments, video recording, measured results.
- `REPORT.md`: outcome summary, recording links, and seven answers.
- `assignment2_answers.json`: machine-readable questions, answers, and measurements.

The older Q/value helper files and CartPole demo are independent of this implementation. GitHub publication is still a separate submission step; no remote repository has been created or uploaded.
