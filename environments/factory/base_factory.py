from typing import List, Union, Iterable

import gym
from gym import spaces
import numpy as np
from pathlib import Path

from environments import helpers as h
from environments.logging.monitor import FactoryMonitor


class AgentState:

    def __init__(self, i: int, action: int):
        self.i = i
        self.action = action

        self.collision_vector = None
        self.action_valid = None
        self.pos = None
        self.info = {}

    @property
    def collisions(self):
        return np.argwhere(self.collision_vector != 0).flatten()

    def update(self, **kwargs):                             # is this hacky?? o.0
        for key, value in kwargs.items():
            if hasattr(self, key):
                self.__setattr__(key, value)
            else:
                raise AttributeError(f'"{key}" cannot be updated, this attr is not a part of {self.__class__.__name__}')


class BaseFactory(gym.Env):

    @property
    def action_space(self):
        return spaces.Discrete(self._registered_actions)

    @property
    def observation_space(self):
        return spaces.Box(low=-1, high=1, shape=self.state.shape, dtype=np.float32)

    @property
    def monitor_as_df_list(self):
        return [x.to_pd_dataframe() for x in self._monitor_list]

    @property
    def movement_actions(self):
        return (int(self.allow_vertical_movement) + int(self.allow_horizontal_movement)) * 4

    @property
    def string_slices(self):
        return {value: key for key, value in self.slice_strings.items()}

    def __init__(self, level='simple', n_agents=1, max_steps=1e3):
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.allow_vertical_movement = True
        self.allow_horizontal_movement = True
        self.allow_no_OP = True
        self._monitor_list = list()
        self._registered_actions = self.movement_actions + int(self.allow_no_OP)
        self.level = h.one_hot_level(
            h.parse_level(Path(__file__).parent / h.LEVELS_DIR / f'{level}.txt')
        )
        self.slice_strings = {0: 'level', **{i: f'agent#{i}' for i in range(1, self.n_agents+1)}}
        self.reset()

    def __init_subclass__(cls):
        print(cls)

    def register_additional_actions(self):
        raise NotImplementedError('Please register additional actions ')

    def reset(self) -> (np.ndarray, int, bool, dict):
        self.done = False
        self.steps = 0
        self.cumulative_reward = 0
        self.monitor = FactoryMonitor(self)
        self._monitor_list.append(self.monitor)
        self.agent_states = []
        # Agent placement ...
        agents = np.zeros((self.n_agents, *self.level.shape), dtype=np.int8)
        floor_tiles = np.argwhere(self.level == h.IS_FREE_CELL)
        # ... on random positions
        np.random.shuffle(floor_tiles)
        for i, (x, y) in enumerate(floor_tiles[:self.n_agents]):
            agents[i, x, y] = h.IS_OCCUPIED_CELL
            agent_state = AgentState(i, -1)
            agent_state.update(pos=[x, y])
            self.agent_states.append(agent_state)
        # state.shape = level, agent 1,..., agent n,
        self.state = np.concatenate((np.expand_dims(self.level, axis=0), agents), axis=0)
        # Returns State

        return self.state

    def additional_actions(self, agent_i: int, action: int) -> ((int, int), bool):
        raise NotImplementedError

    def step(self, actions):
        actions = [actions] if isinstance(actions, int) or np.isscalar(actions) else actions
        assert isinstance(actions, Iterable), f'"actions" has to be in [{int, list}]'
        self.steps += 1

        # Move this in a seperate function?
        states = list()
        for agent_i, action in enumerate(actions):
            agent_i_state = AgentState(agent_i, action)
            if self._is_moving_action(action):
                pos, valid = self.move_or_colide(agent_i, action)
            else:
                pos, valid = self.additional_actions(agent_i, action)
            # Update state accordingly
            agent_i_state.update(pos=pos, action_valid=valid)
            states.append(agent_i_state)

        for i, collision_vec in enumerate(self.check_all_collisions(states, self.state.shape[0])):
            states[i].update(collision_vector=collision_vec)

        self.agent_states = states
        reward, info = self.calculate_reward(states)
        self.cumulative_reward += reward

        if self.steps >= self.max_steps:
            self.done = True
        return self.state, self.cumulative_reward, self.done, info

    def _is_moving_action(self, action):
        if action < self.movement_actions:
            return True
        else:
            return False

    def check_all_collisions(self, agent_states: List[AgentState], collisions: int) -> np.ndarray:
        collision_vecs = np.zeros((len(agent_states), collisions))  # n_agents x n_slices
        for agent_state in agent_states:
            # Register only collisions of moving agents
            if self._is_moving_action(agent_state.action):
                collision_vecs[agent_state.i] = self.check_collisions(agent_state)
        return collision_vecs

    def check_collisions(self, agent_state: AgentState) -> np.ndarray:
        pos_x, pos_y = agent_state.pos
        # FixMe: We need to find a way to spare out some dimensions, eg. an info dimension etc... a[?,]
        collisions_vec = self.state[:, pos_x, pos_y].copy()                 # "vertical fiber" at position of agent i
        collisions_vec[h.AGENT_START_IDX + agent_state.i] = h.IS_FREE_CELL  # no self-collisions
        if agent_state.action_valid:
            # ToDo: Place a function hook here
            pass
        else:
            # Place a marker to indicate a collision with the level boundrys
            collisions_vec[h.LEVEL_IDX] = h.IS_OCCUPIED_CELL
        return collisions_vec

    def do_move(self, agent_i: int, old_pos: (int, int), new_pos: (int, int)) -> None:
        (x, y), (x_new, y_new) = old_pos, new_pos
        self.state[agent_i + h.AGENT_START_IDX, x, y] = h.IS_FREE_CELL
        self.state[agent_i + h.AGENT_START_IDX, x_new, y_new] = h.IS_OCCUPIED_CELL

    def move_or_colide(self, agent_i: int, action: int) -> ((int, int), bool):
        old_pos, new_pos, valid = h.check_agent_move(state=self.state,
                                                     dim=agent_i + h.AGENT_START_IDX,
                                                     action=action)
        if valid:
            # Does not collide width level boundaries
            self.do_move(agent_i, old_pos, new_pos)
            return new_pos, valid
        else:
            # Agent seems to be trying to collide in this step
            return old_pos, valid

    def agent_i_position(self, agent_i: int) -> (int, int):
        positions = np.argwhere(self.state[h.AGENT_START_IDX+agent_i] == h.IS_OCCUPIED_CELL)
        assert positions.shape[0] == 1
        pos_x, pos_y = positions[0]  # a.flatten()
        return pos_x, pos_y

    def free_cells(self, excluded_slices: Union[None, List[int], int] = None) -> np.array:
        excluded_slices = excluded_slices or []
        assert isinstance(excluded_slices, (int, list))
        excluded_slices = excluded_slices if isinstance(excluded_slices, list) else [excluded_slices]

        state = self.state

        if excluded_slices:
            # Todo: Is there a cleaner way?
            inds = list(range(self.state.shape[0]))
            excluded_slices = [inds[x] if x < 0 else x for x in excluded_slices]
            state = self.state[[x for x in inds if x not in excluded_slices]]

        free_cells = np.argwhere(state.sum(0) == h.IS_FREE_CELL)
        np.random.shuffle(free_cells)
        return free_cells

    def calculate_reward(self, agent_states: List[AgentState]) -> (int, dict):
        # Returns: Reward, Info
        # Set to "raise NotImplementedError"
        return 0, {}

    def render(self):
        raise NotImplementedError
