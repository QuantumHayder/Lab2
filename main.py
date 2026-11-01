"""Record initial and learned policies on identical randomly generated mazes."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from gymnasium.wrappers import HumanRendering, RecordVideo, TimeLimit

from GridWorld import GridWorldEnv
from Policy_Iteration import PolicyIteration


def record_policy(policy, initial_observation, size, folder, phase, show, max_steps):
    """Roll out a fixed policy on an exact copy of the original maze."""
    grid = GridWorldEnv(size=size, render_mode="rgb_array")
    env = grid
    try:
        env = RecordVideo(
            TimeLimit(grid, max_episode_steps=max_steps),
            video_folder=str(folder), episode_trigger=lambda episode: True,
            name_prefix=phase, disable_logger=True,
        )
        if show:
            env = HumanRendering(env)
        observation, _ = env.reset(options={
            "agent": initial_observation[:2],
            "target": initial_observation[2:4],
            "bad_cells": [initial_observation[4:6], initial_observation[6:8]],
        })
        total_reward = 0.0
        trajectory = [observation[:2].tolist()]
        actions = []
        while True:
            action = policy[grid.observation_to_state(observation)]
            observation, reward, terminated, truncated, info = env.step(action)
            actions.append(int(action))
            total_reward += reward
            trajectory.append(observation[:2].tolist())
            if terminated or truncated:
                result = {
                    "phase": phase,
                    "initial_observation": list(initial_observation),
                    "steps": len(actions), "total_reward": total_reward,
                    "outcome": info['outcome'] if terminated else 'time_limit',
                    "terminated": terminated, "truncated": truncated,
                    "trajectory": trajectory, "actions": actions,
                    "video": f"{folder.name}/{phase}-episode-0.mp4",
                }
                break
    finally:
        env.close()
    print(f"  {phase}: {result['outcome']}, {result['steps']} steps, return={total_reward}")
    return result


def main(show=False, episodes=3, seed=42, size=5, max_steps=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    max_steps = size * size * 4 if max_steps is None else max_steps
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    video_folder = Path(__file__).resolve().parent / "videos" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    results = []
    grid = GridWorldEnv(size=size)
    try:
        for episode in range(episodes):
            observation, _ = grid.reset(seed=seed + episode)
            initial_observation = observation.tolist()
            solver = PolicyIteration(grid, gamma=0.9)
            print(f"Maze {episode}: seed={seed + episode}, observation={initial_observation}")
            # Record the solver's actual initial policy (right in every state).
            # This rollout does not update values or teach the DP solver.
            before = record_policy(
                solver.policy.copy(), initial_observation, size,
                video_folder / f"maze-{episode}-before-learning",
                "before-learning", show, max_steps,
            )
            iterations = solver.policy_iteration(max_iterations=1000)
            if not solver.converged:
                raise RuntimeError("Policy iteration reached its limit without converging.")
            print(f"  Policy stabilized after {iterations} rounds")
            after = record_policy(
                solver.policy.copy(), initial_observation, size,
                video_folder / f"maze-{episode}-after-learning",
                "after-learning", show, max_steps,
            )
            results.append({
                "episode": episode, "seed": seed + episode, "size": size,
                "initial_observation": initial_observation,
                "initial_policy": "right (action 0) in every nonterminal state",
                "planning_states": len(solver.states), "gamma": solver.gamma,
                "theta": 1e-8, "policy_iterations": iterations,
                "converged": solver.converged, "max_steps": max_steps,
                "before_learning": before, "after_learning": after,
            })
    finally:
        grid.close()
    (video_folder / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Before/after videos and results saved to: {video_folder}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Display both rollouts while recording")
    parser.add_argument("--episodes", type=int, default=3, help="Number of maze comparisons (two videos each)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--max-steps", type=int, help="Step limit for each rollout (default: 4 * size * size)")
    args = parser.parse_args()
    main(show=args.show, episodes=args.episodes, seed=args.seed, size=args.size, max_steps=args.max_steps)
