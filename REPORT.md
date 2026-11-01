# Assignment 2 report

## Implementation and assumptions

The implementation now matches the PDF observation and reward specification: eight integer coordinates for agent, goal, and two bad cells; four deterministic actions; +1 on entering the goal, -1 on entering a bad cell, and 0 otherwise. The PDF does not specify termination on bad cells; this implementation explicitly chooses terminal failure. Four distinct cells are sampled without replacement on reset. Planning enumerates all 25 agent positions for the selected fixed layout and is repeated after every reset. The recording wrapper imposes a 100-step limit on 5x5 rollouts; this external limit is not part of the stationary planning model.

Coordinates use (row, column). Actions are 0=right, 1=up, 2=left, 3=down. Moves beyond the grid boundary stay in place. Terminal states are absorbing with zero subsequent reward. Random layouts are not filtered for solvability. With zero intermediate reward, an optimal policy may avoid failure indefinitely when the goal is unreachable; the demo records this as time_limit, not success.

Policy evaluation applies V(s) = r + gamma * V(next_state), omitting the continuation term on terminal transitions. Policy improvement selects a maximizing action and preserves the current action on numerical ties. The model is cached once per layout. This is model-based dynamic programming; no sampled training episodes are required.

## Measured outcomes

| Seed | Policy rounds | Rollout steps | Return | Outcome |
| --- | --- | --- | --- | --- |
| 42 | 7 | 3 | 1.0 | goal |
| 43 | 7 | 4 | 1.0 | goal |
| 44 | 4 | 4 | 1.0 | goal |

[Raw measured results](videos/20261101-145838-960180/results.json). Gamma=0.9; theta=1e-8; 25 planning states per layout.

## Recorded agent behavior

- [Seed 42 recording](videos/20261101-145838-960180/gridworld-policy-iteration-episode-0.mp4)
- [Seed 43 recording](videos/20261101-145838-960180/gridworld-policy-iteration-episode-1.mp4)
- [Seed 44 recording](videos/20261101-145838-960180/gridworld-policy-iteration-episode-2.mp4)

The videos use PyGame-generated RGB frames at 600x600 and 2 FPS, recorded by Gymnasium RecordVideo. All three files were decoded successfully, with one initial frame plus one frame per step. Earlier recordings in other folders predate the bad-cell implementation.

## Verification

Seven automated tests pass. They check Gymnasium compatibility, seeded distinct placements, observation copies, model/step agreement, boundary and terminal rewards, unreachable goals, validation, rendering of all cell types, and optimal values and paths against independent breadth-first search on 90 layouts (30 seeds each for 2x2, 3x3, and 5x5). Numeric comparisons respect the policy-evaluation tolerance.

## Answers to the seven questions

### 1. What is the state-space size of the 5x5 Grid Maze problem?

The nominal observation space described in the PDF is 5^8 = 25^4 = 390625 combinations: eight coordinates, each with five values. This includes invalid overlapping goal/bad-cell placements. With two ordered bad-cell entries, the number of distinct initial placements is 25 * 24 * 23 * 22 = 303600. If agent overlap with a goal or bad cell is allowed for terminal observations, the modeled state count across all valid layouts is (25 * 24 * 23) * 25 = 345000. Equivalent bad-cell orderings could be merged as an additional optimization. For one fixed layout only the agent position changes, so the implemented solver enumerates 25 states, including the goal and two terminal bad-cell positions. It replans when reset selects another layout.

### 2. How to optimize the policy iteration for the Grid Maze problem?

Keep the goal and bad-cell locations fixed during an episode and solve over the 25 agent positions, then replan or reuse a cached policy after a layout change. Precompute the deterministic successor, reward, and termination flag for each nonterminal state-action pair; there are 22 * 4 = 88 such pairs for a 5x5 layout. A Bellman backup then uses one successor rather than summing over every state. The implementation uses this layout decomposition and model cache, in-place discounted evaluation, zero-valued terminal states, an evaluation tolerance, and stable tie-breaking that retains the previous action when it is already optimal. Further optimizations include NumPy arrays, caching policies by layout, and modified policy iteration with partial evaluation followed by a final convergence check.

### 3. How many iterations did it take to converge on a stable policy for 5x5 maze?

For the complete 5x5 maze with two bad cells, the recorded runs for seeds 42, 43, and 44 converged in 7, 7, and 4 outer policy-iteration rounds respectively. These counts include the final round that confirmed policy stability. The solver used gamma=0.9, theta=1e-8, zero initial values, and an initial right action in each nonterminal state, planning over 25 positions for each layout. The rollouts reached the goal in 3, 4, and 4 steps, respectively, with return +1 and no bad-cell collisions. Outer rounds are distinct from the Bellman sweeps performed during evaluation and the steps taken in the environment. There is no universal count for every 5x5 maze: layout, initialization, discount, tolerance, and update order affect convergence.

### 4. Explain, with an example, how policy iteration behaves with multiple goal cells.

Represent every goal as a terminal state and give its reward when the agent enters it. Policy evaluation computes expected discounted return under the current policy, and improvement chooses the action with the greatest return. With equal +1 goal rewards, zero intermediate rewards, discount 0 < gamma < 1, and no hazards, the optimal policy follows a shortest path to a nearest reachable goal. For example, from agent position (2, 2), goals at (2, 4) and (0, 0) are 2 and 4 steps away. At gamma = 0.9, the corresponding returns are 0.9 and 0.9^3 = 0.729, so the policy selects the first goal. Equally valuable goals can produce multiple optimal actions; retaining the current action on a tie keeps policy improvement stable. Different goal rewards or penalties along a path can make a farther goal preferable. The assignment implementation has one goal; this multiple-goal case is a conceptual extension. Bad-cell terminal failures must also be included in the transition model.

### 5. Can policy iteration work on a 100x100 maze? Explain why?

Yes, with a fixed layout and a known finite transition model. A 100x100 maze has 10000 agent-position states and at most 40000 deterministic state-action pairs, so sparse or cached-model iterative policy evaluation is feasible, though runtime depends on the discount and number of sweeps. The implementation uses this fixed-layout decomposition and can be configured with --size 100, but performance at that size has not been benchmarked. Dense transition matrices and dense linear solves should be avoided. Enumerating all agent, goal, and two-bad-cell configurations instead gives a nominal (10000)^4 = 10^16 states, making exhaustive tabular policy iteration impractical on ordinary hardware. Solving one layout at a time avoids that combinatorial expansion.

### 6. Can policy iteration work on a continuous-space maze? Explain why?

The current tabular policy-iteration algorithm cannot directly handle continuous positions because an uncountably infinite state space cannot be enumerated or stored in a value table. Policy iteration exists mathematically for continuous-state decision processes, but generally requires solving functional equations rather than finite tabular backups. A practical approximation is to discretize positions into grid cells and run tabular policy iteration; finer cells improve spatial precision but increase memory and computation. Another option is approximate policy iteration using a parameterized value function or policy and sampled or numerical evaluation. These methods require additional assumptions and do not automatically inherit the exact convergence guarantees of finite discounted tabular policy iteration.

### 7. Can policy iteration work with moving bad cells (like Packman moving ghosts)? Explain why?

Yes, if the environment is modeled as a finite, fully observed Markov decision process with known, stationary transition probabilities. The state must include the agent position and all ghost positions, plus any ghost direction, movement phase, or other information needed to predict the next state. Each transition must model both the agent movement and ghost movement and define collision rewards and termination, including any crossing-path collisions. For random ghost movement, a Bellman backup uses the probability-weighted sum over all possible next states, rather than the single successor used by the current deterministic solver. The joint state space grows rapidly with the number of ghosts, making exact policy iteration expensive. If ghost positions are hidden, the agent needs a belief-state or other partial-observation approach. If ghosts adapt or act strategically, their behavior must be included in the model or handled using a nonstationary or game-based method. Treating moving ghosts as fixed bad cells would give an incorrect transition model; the implemented bad cells remain stationary during an episode, so moving ghosts would require this model extension.

## Deliverable status

Python code, recorded videos, this report, and machine-readable answers are available locally. A GitHub repository has not been published from this workspace; repository creation/upload remains a submission step. Multiple goals, continuous states, and moving ghosts are discussion questions, not required features of the specified static maze.

## Before/after policy comparison

The demo now records the initial all-right policy before computing policy iteration, then records the converged policy from the exact same initial observation. A seed-42 comparison with a 20-step limit produced a boundary loop before learning (20 steps, return 0, time-limit truncation), followed by goal success after learning (3 steps, return +1, 7 policy-iteration rounds). Both videos were decoded and their initial frames matched. The initial rollout is for comparison and does not provide training data to the DP solver.

- [before-learning video](videos/20261101-150351-942190/maze-0-before-learning/before-learning-episode-0.mp4)
- [after-learning video](videos/20261101-150351-942190/maze-0-after-learning/after-learning-episode-0.mp4)
- [Paired results](videos/20261101-150351-942190/results.json)
