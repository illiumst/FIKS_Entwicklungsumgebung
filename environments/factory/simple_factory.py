from collections import OrderedDict
from dataclasses import dataclass
from typing import List
import random

import numpy as np

from environments.factory.base_factory import BaseFactory, AgentState
from environments import helpers as h

from environments.factory.renderer import Renderer
from environments.factory.renderer import Entity
from environments.logging.monitor import MonitorCallback

DIRT_INDEX = -1


@dataclass
class DirtProperties:
    clean_amount = 10
    max_spawn_ratio = 0.1
    gain_amount = 0.1
    spawn_frequency = 5


class SimpleFactory(BaseFactory):

    def register_additional_actions(self):
        return 1

    def _is_clean_up_action(self, action):
        return self.action_space.n - 1 == action

    def __init__(self, *args, dirt_properties: DirtProperties, verbose=False, **kwargs):
        self._dirt_properties = dirt_properties
        self.verbose = verbose
        super(SimpleFactory, self).__init__(*args, **kwargs)
        self.slice_strings.update({self.state.shape[0]-1: 'dirt'})
        self.renderer = None  # expensive - dont use it when not required !

    def render(self):
        if not self.renderer:  # lazy init
            height, width = self.state.shape[1:]
            self.renderer = Renderer(width, height, view_radius=2)

        dirt      = [Entity('dirt', [x, y], min(0.15+self.state[DIRT_INDEX, x, y], 1.5), 'scale')
                     for x, y in np.argwhere(self.state[DIRT_INDEX] > h.IS_FREE_CELL)]
        walls     = [Entity('wall', pos) for pos in np.argwhere(self.state[h.LEVEL_IDX] > h.IS_FREE_CELL)]

        def asset_str(agent):
            cols = ' '.join([self.slice_strings[j] for j in agent.collisions])
            if 'agent' in cols:
                return 'agent_collision'
            elif not agent.action_valid or 'level' in cols or 'agent' in cols:
                return f'agent{agent.i + 1}violation'
            elif self._is_clean_up_action(agent.action):
                return f'agent{agent.i + 1}valid'
            else:
                return f'agent{agent.i + 1}'

        agents = {f'agent{i+1}': [Entity(asset_str(agent), agent.pos)]
                  for i, agent in enumerate(self.agent_states)}
        self.renderer.render(OrderedDict(dirt=dirt, wall=walls, **agents))

    def spawn_dirt(self) -> None:
        free_for_dirt = self.free_cells(excluded_slices=DIRT_INDEX)

        # randomly distribute dirt across the grid
        n_dirt_tiles = int(random.uniform(0, self._dirt_properties.max_spawn_ratio) * len(free_for_dirt))
        for x, y in free_for_dirt[:n_dirt_tiles]:
            self.state[DIRT_INDEX, x, y] += self._dirt_properties.gain_amount

    def clean_up(self, pos: (int, int)) -> ((int, int), bool):
        new_dirt_amount = self.state[DIRT_INDEX][pos] - self._dirt_properties.clean_amount
        cleanup_was_sucessfull: bool
        if self.state[DIRT_INDEX][pos] == h.IS_FREE_CELL:
            cleanup_was_sucessfull = False
            return pos, cleanup_was_sucessfull
        else:
            cleanup_was_sucessfull = True
            self.state[DIRT_INDEX][pos] = max(new_dirt_amount, h.IS_FREE_CELL)
            return pos, cleanup_was_sucessfull

    def step(self, actions):
        if self.state[h.LEVEL_IDX][self.agent_i_position(0)] == h.IS_OCCUPIED_CELL:
            print(f'fAgent placed on wall!!!!, step is :{self.steps}')
            raise Exception('Agent placed on wall!!!!')
        _, _, _, info = super(SimpleFactory, self).step(actions)
        if not self.next_dirt_spawn:
            self.spawn_dirt()
            self.next_dirt_spawn = self._dirt_properties.spawn_frequency
        else:
            self.next_dirt_spawn -= 1
        return self.state, self.cumulative_reward, self.done, info

    def additional_actions(self, agent_i: int, action: int) -> ((int, int), bool):
        if action != self._is_moving_action(action):
            if self._is_clean_up_action(action):
                agent_i_pos = self.agent_i_position(agent_i)
                _, valid = self.clean_up(agent_i_pos)
                return agent_i_pos, valid
            else:
                raise RuntimeError('This should not happen!!!')
        else:
            raise RuntimeError('This should not happen!!!')

    def reset(self) -> (np.ndarray, int, bool, dict):
        _ = super().reset()  # state, reward, done, info ... =
        dirt_slice = np.zeros((1, *self.state.shape[1:]))
        self.state = np.concatenate((self.state, dirt_slice))  # dirt is now the last slice
        self.spawn_dirt()
        self.next_dirt_spawn = self._dirt_properties.spawn_frequency
        return self.state

    def calculate_reward(self, agent_states: List[AgentState]) -> (int, dict):
        # TODO: What reward to use?
        current_dirt_amount = self.state[DIRT_INDEX].sum()
        dirty_tiles = len(np.nonzero(self.state[DIRT_INDEX]))

        try:
            # penalty = current_dirt_amount
            penalty = 0
        except (ZeroDivisionError, RuntimeWarning):
            penalty = 0
        inforcements = 0
        for agent_state in agent_states:
            cols = agent_state.collisions
            self.print(f't = {self.steps}\tAgent {agent_state.i} has collisions with '
                       f'{[self.slice_strings[entity] for entity in cols if entity != self.string_slices["dirt"]]}')
            if self._is_clean_up_action(agent_state.action):
                if agent_state.action_valid:
                    inforcements += 10
                    self.print(f'Agent {agent_state.i} did just clean up some dirt at {agent_state.pos}.')
                    self.monitor.add('dirt_cleaned', self._dirt_properties.clean_amount)
                else:
                    self.print(f'Agent {agent_state.i} just tried to clean up some dirt '
                               f'at {agent_state.pos}, but was unsucsessfull.')
                    self.monitor.add('failed_cleanup_attempt', 1)
            elif self._is_moving_action(agent_state.action):
                if not agent_state.action_valid:
                    penalty += 10
                else:
                    inforcements += 1

            for entity in cols:
                if entity != self.string_slices["dirt"]:
                    self.monitor.add(f'agent_{agent_state.i}_vs_{self.slice_strings[entity]}', 1)

        this_step_reward = max(0, inforcements-penalty)
        self.monitor.set('dirt_amount', current_dirt_amount)
        self.monitor.set('dirty_tiles', dirty_tiles)
        self.print(f"reward is {this_step_reward}")
        return this_step_reward, {}

    def print(self, string):
        if self.verbose:
            print(string)


if __name__ == '__main__':
    render = True

    dirt_props = DirtProperties()
    factory = SimpleFactory(n_agents=2, dirt_properties=dirt_props)
    with MonitorCallback(factory):
        for epoch in range(100):
            random_actions = [(random.randint(0, 8), random.randint(0, 8)) for _ in range(200)]
            env_state, reward, done_bool, _ = factory.reset()
            for agent_i_action in random_actions:
                env_state, reward, done_bool, info_obj = factory.step(agent_i_action)
                if render:
                    factory.render()
                if done_bool:
                    break
            print(f'Factory run {epoch} done, reward is:\n    {reward}')
