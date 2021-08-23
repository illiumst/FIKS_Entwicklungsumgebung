import abc
import time
from pathlib import Path
from typing import List, Union, Iterable

import gym
import numpy as np
from gym import spaces

import yaml
from gym.wrappers import FrameStack

from environments.factory.base.shadow_casting import Map
from environments.factory.renderer import Renderer, RenderEntity
from environments.helpers import Constants as c, Constants
from environments import helpers as h
from environments.factory.base.objects import Slice, Agent, Tile, Action
from environments.factory.base.registers import StateSlices, Actions, Entities, Agents, Doors, FloorTiles
from environments.utility_classes import MovementProperties

REC_TAC = 'rec'


# noinspection PyAttributeOutsideInit
class BaseFactory(gym.Env):

    @property
    def action_space(self):
        return spaces.Discrete(self._actions.n)

    @property
    def observation_space(self):
        slices = self._slices.n_observable_slices
        level_shape = (self.pomdp_r * 2 + 1, self.pomdp_r * 2 + 1) if self.pomdp_r else self._level_shape
        space = spaces.Box(low=0, high=1, shape=(slices, *level_shape), dtype=np.float32)
        return space

    @property
    def pomdp_diameter(self):
        return self.pomdp_r * 2 + 1

    @property
    def movement_actions(self):
        return self._actions.movement_actions

    def __enter__(self):
        return self if self.frames_to_stack == 0 else FrameStack(self, self.frames_to_stack)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __init__(self, level_name='simple', n_agents=1, max_steps=int(5e2), pomdp_r: Union[None, int] = 0,
                 movement_properties: MovementProperties = MovementProperties(), parse_doors=False,
                 combin_agent_slices_in_obs: bool = False, frames_to_stack=0, record_episodes=False,
                 omit_agent_slice_in_obs=False, done_at_collision=False, cast_shadows=True,
                 verbose=False, doors_have_area=True, env_seed=time.time_ns(), **kwargs):
        assert frames_to_stack != 1 and frames_to_stack >= 0, "'frames_to_stack' cannot be negative or 1."

        # Attribute Assignment
        self.env_seed = env_seed
        self._base_rng = np.random.default_rng(self.env_seed)
        self.movement_properties = movement_properties
        self.level_name = level_name
        self._level_shape = None
        self.verbose = verbose
        self._renderer = None  # expensive - don't use it when not required !

        self.n_agents = n_agents

        self.max_steps = max_steps
        self.pomdp_r = pomdp_r
        self.combin_agent_slices_in_obs = combin_agent_slices_in_obs
        self.omit_agent_slice_in_obs = omit_agent_slice_in_obs
        self.cast_shadows = cast_shadows
        self.frames_to_stack = frames_to_stack

        self.done_at_collision = done_at_collision
        self.record_episodes = record_episodes
        self.parse_doors = parse_doors
        self.doors_have_area = doors_have_area

        # Actions
        self._actions = Actions(self.movement_properties, can_use_doors=self.parse_doors)
        if additional_actions := self.additional_actions:
            self._actions.register_additional_items(additional_actions)

        # Reset
        self.reset()

    def _init_state_slices(self) -> StateSlices:
        state_slices = StateSlices()

        # Objects
        # Level
        level_filepath = Path(__file__).parent.parent / h.LEVELS_DIR / f'{self.level_name}.txt'
        parsed_level = h.parse_level(level_filepath)
        level = [Slice(c.LEVEL, h.one_hot_level(parsed_level), is_blocking_light=True)]
        self._level_shape = level[0].shape

        # Doors
        parsed_doors = h.one_hot_level(parsed_level, c.DOOR)
        if parsed_doors.any():
            doors = [Slice(c.DOORS, parsed_doors, is_blocking_light=True)]
        else:
            doors = []

        # Agents
        agents = []
        agent_names = [f'{c.AGENT.value}#{i}' for i in range(self.n_agents)]

        if self.combin_agent_slices_in_obs and self.omit_agent_slice_in_obs:
            if self.n_agents == 1:
                observables = [False]
            else:
                observables = [True] + ([False] * (self.n_agents - 1))
        elif self.combin_agent_slices_in_obs and not self.omit_agent_slice_in_obs:
            observables = [True] + ([False] * (self.n_agents - 1))
        elif not self.combin_agent_slices_in_obs and self.omit_agent_slice_in_obs:
            observables = [False] + ([True] * (self.n_agents - 1))
        elif not self.combin_agent_slices_in_obs and not self.omit_agent_slice_in_obs:
            observables = [True] * self.n_agents
        else:
            raise RuntimeError('This should not happen!')

        for observable, agent_name in zip(observables, agent_names):
            agents.append(Slice(agent_name, np.zeros_like(level[0].slice, dtype=np.float32), is_observable=observable))
        state_slices.register_additional_items(level+doors+agents+self.additional_slices)
        return state_slices

    def _init_obs_cube(self) -> np.ndarray:
        x, y = self._slices.by_enum(c.LEVEL).shape
        state = np.zeros((len(self._slices), x, y), dtype=np.float32)
        state[0] = self._slices.by_enum(c.LEVEL).slice
        if r := self.pomdp_r:
            self._padded_obs_cube = np.full((len(self._slices), x + r*2, y + r*2), c.FREE_CELL.value, dtype=np.float32)
            self._padded_obs_cube[0] = c.OCCUPIED_CELL.value
            self._padded_obs_cube[:, r:r+x, r:r+y] = state
        if self.combin_agent_slices_in_obs and self.n_agents > 1:
            self._combined_obs_cube = np.zeros(self.observation_space.shape, dtype=np.float32)
        return state

    def _init_entities(self):
        # Tile Init
        self._tiles = FloorTiles.from_argwhere_coordinates(self._slices.by_enum(c.LEVEL).free_tiles)

        # Door Init
        if self.parse_doors:
            tiles = [self._tiles.by_pos(x) for x in self._slices.by_enum(c.DOORS).occupied_tiles]
            self._doors = Doors.from_tiles(tiles, context=self._tiles, has_area=self.doors_have_area)

        # Agent Init on random positions
        self._agents = Agents.from_tiles(self._base_rng.choice(self._tiles, self.n_agents))
        entities = Entities()
        entities.register_additional_items([self._agents])

        if self.parse_doors:
            entities.register_additional_items([self._doors])

        if additional_entities := self.additional_entities:
            entities.register_additional_items(additional_entities)

        return entities

    def reset(self) -> (np.ndarray, int, bool, dict):
        self._slices = self._init_state_slices()
        self._obs_cube = self._init_obs_cube()
        self._entitites = self._init_entities()
        self.do_additional_reset()
        self._flush_state()
        self._steps = 0

        obs = self._get_observations()
        return obs

    def step(self, actions):
        actions = [actions] if isinstance(actions, int) or np.isscalar(actions) else actions
        assert isinstance(actions, Iterable), f'"actions" has to be in [{int, list}]'
        self._steps += 1
        done = False

        # Pre step Hook for later use
        self.hook_pre_step()

        # Move this in a seperate function?
        for action, agent in zip(actions, self._agents):
            agent.clear_temp_sate()
            action_obj = self._actions[action]
            if self._actions.is_moving_action(action_obj):
                valid = self._move_or_colide(agent, action_obj)
            elif self._actions.is_no_op(action_obj):
                valid = c.VALID.value
            elif self._actions.is_door_usage(action_obj):
                valid = self._handle_door_interaction(agent)
            else:
                valid = self.do_additional_actions(agent, action_obj)
            assert valid is not None, 'This should not happen, every Action musst be detected correctly!'
            agent.temp_action = action_obj
            agent.temp_valid = valid

        # In-between step Hook for later use
        info = self.do_additional_step()

        # Write to observation cube
        self._flush_state()

        tiles_with_collisions = self.get_all_tiles_with_collisions()
        for tile in tiles_with_collisions:
            guests = tile.guests_that_can_collide
            for i, guest in enumerate(guests):
                this_collisions = guests[:]
                del this_collisions[i]
                guest.temp_collisions = this_collisions

        if self.done_at_collision and tiles_with_collisions:
            done = True

        # Step the door close intervall
        if self.parse_doors:
            self._doors.tick_doors()

        # Finalize
        reward, reward_info = self.calculate_reward()
        info.update(reward_info)
        if self._steps >= self.max_steps:
            done = True
        info.update(step_reward=reward, step=self._steps)
        if self.record_episodes:
            info.update(self._summarize_state())

        # Post step Hook for later use
        info.update(self.hook_post_step())

        obs = self._get_observations()

        return obs, reward, done, info

    def _handle_door_interaction(self, agent):
        # Check if agent really is standing on a door:
        if self.doors_have_area:
            door = self._doors.get_near_position(agent.pos)
        else:
            door = self._doors.by_pos(agent.pos)
        if door is not None:
            door.use()
            return c.VALID.value
        # When he doesn't...
        else:
            return c.NOT_VALID.value

    def _flush_state(self):
        self._obs_cube[np.arange(len(self._slices)) != self._slices.get_idx(c.LEVEL)] = c.FREE_CELL.value
        if self.parse_doors:
            for door in self._doors:
                if door.is_open and self._obs_cube[self._slices.get_idx(c.DOORS)][door.pos] != c.OPEN_DOOR.value:
                    self._obs_cube[self._slices.get_idx(c.DOORS)][door.pos] = c.OPEN_DOOR.value
                elif door.is_closed and self._obs_cube[self._slices.get_idx(c.DOORS)][door.pos] != c.CLOSED_DOOR.value:
                    self._obs_cube[self._slices.get_idx(c.DOORS)][door.pos] = c.CLOSED_DOOR.value
        for agent in self._agents:
            self._obs_cube[self._slices.get_idx_by_name(agent.name)][agent.pos] = c.OCCUPIED_CELL.value
            if agent.last_pos != c.NO_POS:
                self._obs_cube[self._slices.get_idx_by_name(agent.name)][agent.last_pos] = c.FREE_CELL.value

    def _get_observations(self) -> np.ndarray:
        if self.n_agents == 1:
            obs = self._build_per_agent_obs(self._agents[0])
        elif self.n_agents >= 2:
            obs = np.stack([self._build_per_agent_obs(agent) for agent in self._agents])
        else:
            raise ValueError('n_agents cannot be smaller than 1!!')
        return obs

    def _build_per_agent_obs(self, agent: Agent) -> np.ndarray:
        first_agent_slice = self._slices.AGENTSTARTIDX
        if r := self.pomdp_r:
            x, y = self._level_shape
            self._padded_obs_cube[:, r:r + x, r:r + y] = self._obs_cube
            global_x, global_y = agent.pos
            global_x += r
            global_y += r
            x0, x1 = max(0, global_x - self.pomdp_r), global_x + self.pomdp_r + 1
            y0, y1 = max(0, global_y - self.pomdp_r), global_y + self.pomdp_r + 1
            obs = self._padded_obs_cube[:, x0:x1, y0:y1]
        else:
            obs = self._obs_cube

        if self.cast_shadows:
            obs_block_light = [obs[idx] != c.OCCUPIED_CELL.value for idx, obs_slice
                               in enumerate(self._slices) if obs_slice.is_blocking_light]
            door_shadowing = False
            if door := self._doors.by_pos(agent.pos):
                if door.is_closed:
                    for group in door.connectivity_subgroups:
                        if agent.last_pos not in group:
                            door_shadowing = True
                            if self.pomdp_r:
                                blocking = [tuple(np.subtract(x, agent.pos) + (self.pomdp_r, self.pomdp_r))
                                            for x in group]
                                xs, ys = zip(*blocking)
                            else:
                                xs, ys = zip(*group)
                            # noinspection PyTypeChecker
                            obs_block_light[self._slices.get_idx(c.LEVEL)][xs, ys] = False

            light_block_map = Map((np.prod(obs_block_light, axis=0) != True).astype(int))
            if self.pomdp_r:
                light_block_map = light_block_map.do_fov(self.pomdp_r, self.pomdp_r, max(self._level_shape))
            else:
                light_block_map = light_block_map.do_fov(*agent.pos, max(self._level_shape))
            if door_shadowing:
                # noinspection PyUnboundLocalVariable
                light_block_map[xs, ys] = 0
            agent.temp_light_map = light_block_map
            for obs_idx in range(obs.shape[0]):
                if self._slices[obs_idx].can_be_shadowed:
                    obs[obs_idx] = (obs[obs_idx] * light_block_map) - (
                            (1 - light_block_map) * obs[self._slices.get_idx(c.LEVEL)]
                    )

        if self.combin_agent_slices_in_obs and self.n_agents > 1:
            agent_obs = np.sum(obs[[key for key, l_slice in self._slices.items() if c.AGENT.name in l_slice.name and
                                    (not self.omit_agent_slice_in_obs and l_slice.name != agent.name)]],
                               axis=0, keepdims=True)
            obs = np.concatenate((obs[:first_agent_slice], agent_obs, obs[first_agent_slice+self.n_agents:]))
            return obs
        else:
            if self.omit_agent_slice_in_obs:
                obs_new = obs[[key for key, val in self._slices.items() if val.name != agent.name]]
                return obs_new
            else:
                return obs

    def get_all_tiles_with_collisions(self) -> List[Tile]:
        tiles_with_collisions = list()
        for tile in self._tiles:
            if tile.is_occupied():
                guests = [guest for guest in tile.guests if guest.can_collide]
                if len(guests) >= 2:
                    tiles_with_collisions.append(tile)
        return tiles_with_collisions

    def _move_or_colide(self, agent: Agent, action: Action) -> Constants:
        new_tile, valid = self._check_agent_move(agent, action)
        if valid:
            # Does not collide width level boundaries
            return agent.move(new_tile)
        else:
            # Agent seems to be trying to collide in this step
            return c.NOT_VALID

    def _check_agent_move(self, agent, action: Action) -> (Tile, bool):
        # Actions
        x_diff, y_diff = h.ACTIONMAP[action.name]
        x_new = agent.x + x_diff
        y_new = agent.y + y_diff

        new_tile = self._tiles.by_pos((x_new, y_new))
        if new_tile:
            valid = c.VALID
        else:
            tile = agent.tile
            valid = c.VALID
            return tile, valid

        if self.parse_doors and agent.last_pos != c.NO_POS:
            if door := self._doors.by_pos(new_tile.pos):
                if door.can_collide:
                    return agent.tile, c.NOT_VALID
                else:  # door.is_closed:
                    pass

            if door := self._doors.by_pos(agent.pos):
                if door.is_open:
                    pass
                else:  # door.is_closed:
                    if door.is_linked(agent.last_pos, new_tile.pos):
                        pass
                    else:
                        return agent.tile, c.NOT_VALID
            else:
                pass
        else:
            pass

        return new_tile, valid

    def calculate_reward(self) -> (int, dict):
        # Returns: Reward, Info
        info_dict = dict()
        reward = 0

        for agent in self._agents:
            if self._actions.is_moving_action(agent.temp_action):
                if agent.temp_valid:
                    # info_dict.update(movement=1)
                    reward -= 0.00
                else:
                    # self.print('collision')
                    reward -= 0.01
                    self.print(f'{agent.name} just hit the wall at {agent.pos}.')
                    info_dict.update({f'{agent.name}_vs_LEVEL': 1})

            elif self._actions.is_door_usage(agent.temp_action):
                if agent.temp_valid:
                    self.print(f'{agent.name} did just use the door at {agent.pos}.')
                    info_dict.update(door_used=1)
                else:
                    reward -= 0.01
                    self.print(f'{agent.name} just tried to use a door at {agent.pos}, but failed.')
                    info_dict.update({f'{agent.name}_failed_action': 1})
                    info_dict.update({f'{agent.name}_failed_door_open': 1})
            elif self._actions.is_no_op(agent.temp_action):
                info_dict.update(no_op=1)
                reward -= 0.00

            additional_reward, additional_info_dict = self.calculate_additional_reward(agent)
            reward += additional_reward
            info_dict.update(additional_info_dict)

            for other_agent in agent.temp_collisions:
                info_dict.update({f'{agent.name}_vs_{other_agent.name}': 1})

        self.print(f"reward is {reward}")
        return reward, info_dict

    def render(self, mode='human'):
        if not self._renderer:  # lazy init
            height, width = self._obs_cube.shape[1:]
            self._renderer = Renderer(width, height, view_radius=self.pomdp_r, fps=5)

        walls = [RenderEntity('wall', pos)
                 for pos in np.argwhere(self._slices.by_enum(c.LEVEL).slice == c.OCCUPIED_CELL.value)]

        agents = []
        for i, agent in enumerate(self._agents):
            name, state = h.asset_str(agent)
            agents.append(RenderEntity(name, agent.pos, 1, 'none', state, i + 1, agent.temp_light_map))
        doors = []
        if self.parse_doors:
            for i, door in enumerate(self._doors):
                name, state = 'door_open' if door.is_open else 'door_closed', 'blank'
                doors.append(RenderEntity(name, door.pos, 1, 'none', state, i + 1))
        additional_assets = self.render_additional_assets()

        self._renderer.render(walls + doors + additional_assets + agents)

    def save_params(self, filepath: Path):
        # noinspection PyProtectedMember
        # d = {key: val._asdict() if hasattr(val, '_asdict') else val for key, val in self.__dict__.items()
        d = {key: val for key, val in self.__dict__.items() if not key.startswith('_') and not key.startswith('__')}
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open('w') as f:
            yaml.dump(d, f)
            # pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _summarize_state(self):
        summary = {f'{REC_TAC}_step': self._steps}
        for entity in self._entitites:
            if hasattr(entity, 'summarize_state'):
                summary.update({f'{REC_TAC}_{entity.name}': entity.summarize_state()})
        return summary

    def print(self, string):
        if self.verbose:
            print(string)

    # Properties which are called by the base class to extend beyond attributes of the base class
    @property
    def additional_actions(self) -> Union[Action, List[Action]]:
        """
        When heriting from this Base Class, you musst implement this methode!!!

        :return:            A list of Actions-object holding all additional actions.
        :rtype:             List[Action]
        """
        return []

    @property
    def additional_entities(self) -> Union[Entities, List[Entities]]:
        """
        When heriting from this Base Class, you musst implement this methode!!!

        :return:            A single Entites collection or a list of such.
        :rtype:             Union[Entities, List[Entities]]
        """
        return []

    @property
    def additional_slices(self) -> Union[Slice, List[Slice]]:
        """
        When heriting from this Base Class, you musst implement this methode!!!

        :return:            A list of Slice-objects.
        :rtype:             List[Slice]
        """
        return []

    # Functions which provide additions to functions of the base class
    #  Always call super!!!!!!
    @abc.abstractmethod
    def do_additional_reset(self) -> None:
        pass

    @abc.abstractmethod
    def do_additional_step(self) -> dict:
        return {}

    @abc.abstractmethod
    def do_additional_actions(self, agent: Agent, action: int) -> Union[None, bool]:
        return None

    @abc.abstractmethod
    def calculate_additional_reward(self, agent: Agent) -> (int, dict):
        return 0, {}

    @abc.abstractmethod
    def render_additional_assets(self):
        return []

    # Hooks for in between operations.
    #  Always call super!!!!!!
    @abc.abstractmethod
    def hook_pre_step(self) -> None:
        pass

    @abc.abstractmethod
    def hook_post_step(self) -> dict:
        return {}
