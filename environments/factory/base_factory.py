import numpy as np
from pathlib import Path
from environments import helpers as h


class BaseFactory(object):
    LEVELS_DIR = 'levels'

    def __init__(self, level='simple', n_agents=1, max_steps=1e3):
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.level = h.one_hot_level(
            h.parse_level(Path(__file__).parent / self.LEVELS_DIR / f'{level}.txt')
        )#[np.newaxis, ...]
        self.reset()

    def reset(self):
        self.done = False
        self.steps = 0
        self.agents = np.zeros((self.n_agents, *self.level.shape), dtype=np.int8)
        free_cells = np.argwhere(self.level == 0)
        np.random.shuffle(free_cells)
        for i in range(self.n_agents):
            r, c = free_cells[i]
            self.agents[i, r, c] = 1
        free_cells = free_cells[self.n_agents:]
        self.state = np.concatenate((self.level[np.newaxis, ...], self.agents), 0)
        return self.state, 0, self.done, {}

    def step(self, actions):
        assert type(actions) in [int, list]
        if type(actions) == int:
            actions = [actions]
        self.steps += 1
        r = 0
        # level, agent 1,..., agent n,
        for i, a in enumerate(actions):
            old_pos, new_pos, valid = h.check_agent_move(state=self.state, dim=i+1, action=a)
            if valid:
                self.make_move(i, old_pos, new_pos)
        collision_vecs = []
        for i in range(self.n_agents):  # might as well save the positions (redundant)
            agent_slice = self.state[i+1]
            x, y = np.argwhere(agent_slice == 1)[0]
            collisions_vec = self.state[:, x, y].copy()  # otherwise you overwrite the grid/state
            collisions_vec[i+1] = 0  # no self-collisions
            collision_vecs.append(collisions_vec)
        self.handle_collisions(collisions_vec)
        r += self.step_core(collisions_vec, actions, r)
        if self.steps >= self.max_steps:
            self.done = True
        return self.state, r, self.done, {}

    def make_move(self, agent_i, old_pos, new_pos):
        (x, y), (x_new, y_new) = old_pos, new_pos
        self.state[agent_i+1, x, y] = 0
        self.state[agent_i+1, x_new, y_new] = 1

    def handle_collisions(self, vecs):
        pass

    def step_core(self, collisions_vec, actions, r):
        return 0
