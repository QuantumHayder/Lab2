from typing import Optional
from itertools import product
import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, size: int = 5, render_mode=None):
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.render_mode = render_mode
        self.window_size = 600
        self.window = None
        self.clock = None
        self._window_closed = False
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 2:
            raise ValueError("size must be an integer >= 2")
        # The size of the square grid (5x5 by default)
        self.size = size

        # Observation order: agent, goal, bad cell 1, bad cell 2 (row, column).
        self._agent_location = np.full(2, -1, dtype=np.int64)
        self._target_location = np.full(2, -1, dtype=np.int64)
        self._bad_locations = np.full((2, 2), -1, dtype=np.int64)
        self.observation_space = gym.spaces.MultiDiscrete([size] * 8)

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([0, 1]),   # Move right (column + 1)
            1: np.array([-1, 0]),  # Move up (row - 1)
            2: np.array([0, -1]),  # Move left (column - 1)
            3: np.array([1, 0]),   # Move down (row + 1)
        }
        
    def _get_obs(self):
        return np.concatenate((self._agent_location, self._target_location,
                               self._bad_locations.ravel())).astype(np.int64)

    def get_states(self):
        """Enumerate agent positions for the layout selected by reset().

        The goal and bad cells stay fixed within an episode. Construct a new
        PolicyIteration solver after resetting to a different layout.
        """
        if np.any(self._target_location < 0):
            raise gym.error.ResetNeeded("Call reset() before constructing a policy.")
        layout = self.observation_to_state(self._get_obs())[2:]
        return ((row, col, *layout) for row, col in product(range(self.size), repeat=2))

    @staticmethod
    def _outcome(state):
        agent = tuple(state[:2])
        if agent == tuple(state[2:4]):
            return "goal"
        if agent in (tuple(state[4:6]), tuple(state[6:8])):
            return "bad_cell"
        return "ongoing"

    def get_actions(self, state):
        return tuple(range(4)) if self._outcome(state) == "ongoing" else ()

    @staticmethod
    def observation_to_state(observation):
        return tuple(int(x) for x in observation)

    def _get_info(self):
        state = self.observation_to_state(self._get_obs())
        return {
            "distance": int(np.abs(self._agent_location - self._target_location).sum()),
            "outcome": self._outcome(state),
            "is_success": self._outcome(state) == "goal",
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Place four distinct objects randomly, or use a supplied test layout.

        options may provide all of agent (2,), target (2,), bad_cells (2, 2).
        Bad cells terminate the episode with -1; the goal terminates with +1.
        Random layouts are not filtered for reachability.
        """
        super().reset(seed=seed)
        if options:
            if set(options) != {"agent", "target", "bad_cells"}:
                raise ValueError("options must contain agent, target, and bad_cells")
            agent = np.asarray(options["agent"])
            target = np.asarray(options["target"])
            bad_cells = np.asarray(options["bad_cells"])
            if agent.shape != (2,) or target.shape != (2,) or bad_cells.shape != (2, 2):
                raise ValueError("Expected agent/target shape (2,) and bad_cells shape (2, 2)")
            positions = np.vstack((agent, target, bad_cells))
            if not np.issubdtype(positions.dtype, np.integer) or np.any(
                (positions < 0) | (positions >= self.size)
            ):
                raise ValueError("Positions must be integer coordinates inside the grid")
            if len(set(map(tuple, positions))) != 4:
                raise ValueError("The agent, goal, and bad cells must start in distinct cells")
        else:
            cells = self.np_random.choice(self.size * self.size, size=4, replace=False)
            positions = np.column_stack((cells // self.size, cells % self.size))
        self._agent_location = positions[0].astype(np.int64).copy()
        self._target_location = positions[1].astype(np.int64).copy()
        self._bad_locations = positions[2:].astype(np.int64).copy()
        self._window_closed = False
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info()

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        if np.any(self._agent_location < 0):
            raise gym.error.ResetNeeded("Call reset() before step().")
        state = self.observation_to_state(self._get_obs())
        next_state, reward, terminated = self.transition(state, action)
        self._agent_location = np.array(next_state[:2], dtype=int)
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self):
        """Draw with PyGame; RGB frames use (height, width, channel) order."""
        if self.render_mode is None:
            return None
        if np.any(self._agent_location < 0):
            raise gym.error.ResetNeeded("Call reset() before render().")

        import pygame

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((245, 247, 250))
        cell_size = self.window_size / self.size
        row, col = self._target_location
        pygame.draw.rect(canvas, (50, 180, 100), pygame.Rect(
            round(col * cell_size), round(row * cell_size),
            round(cell_size), round(cell_size)
        ))
        for row, col in self._bad_locations:
            left, top = round(col * cell_size), round(row * cell_size)
            pygame.draw.rect(canvas, (215, 65, 65),
                             pygame.Rect(left, top, round(cell_size), round(cell_size)))
            inset = cell_size * 0.25
            pygame.draw.line(canvas, (255, 255, 255),
                             (left + inset, top + inset),
                             (left + cell_size - inset, top + cell_size - inset), 5)
            pygame.draw.line(canvas, (255, 255, 255),
                             (left + inset, top + cell_size - inset),
                             (left + cell_size - inset, top + inset), 5)
        row, col = self._agent_location
        pygame.draw.circle(canvas, (45, 105, 220),
                           (round((col + 0.5) * cell_size), round((row + 0.5) * cell_size)),
                           max(1, round(cell_size * 0.3)))
        for index in range(self.size + 1):
            position = min(round(index * cell_size), self.window_size - 1)
            pygame.draw.line(canvas, (65, 75, 90),
                             (0, position), (self.window_size - 1, position), 2)
            pygame.draw.line(canvas, (65, 75, 90),
                             (position, 0), (position, self.window_size - 1), 2)

        if self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2)).copy()

        if self._window_closed:
            return None
        if self.window is None:
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("GridWorld: blue agent, green goal, red bad cells")
            self.clock = pygame.time.Clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                self._window_closed = True
                return None
        self.window.blit(canvas, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            self.window = None
            self.clock = None
    
    def transition(self, state, action):
        """Pure deterministic model shared by dynamic programming and step()."""
        if not self.action_space.contains(action):
            raise ValueError("action must be an integer in [0, 3]")
        if len(state) != 8 or any(
            not isinstance(x, (int, np.integer)) or not 0 <= x < self.size
            for x in state
        ):
            raise ValueError("state must contain eight integer grid coordinates")
        state = tuple(int(x) for x in state)
        if len({state[2:4], state[4:6], state[6:8]}) != 3:
            raise ValueError("Goal and bad cells must be distinct")
        if self._outcome(state) != "ongoing":
            return state, 0.0, True
        next_agent = np.clip(np.array(state[:2]) + self._action_to_direction[action],
                             0, self.size - 1)
        next_state = (int(next_agent[0]), int(next_agent[1]), *state[2:])
        outcome = self._outcome(next_state)
        reward = {"goal": 1.0, "bad_cell": -1.0, "ongoing": 0.0}[outcome]
        return next_state, reward, outcome != "ongoing"
