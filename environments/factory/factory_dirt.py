import time
from pathlib import Path
from typing import List, Union, NamedTuple, Dict
import random

import numpy as np

from algorithms.TSP_dirt_agent import TSPDirtAgent
from environments.helpers import Constants as BaseConstants
from environments.helpers import EnvActions as BaseActions

from environments.factory.base.base_factory import BaseFactory
from environments.factory.base.objects import Agent, Action, Entity, Floor
from environments.factory.base.registers import Entities, EntityRegister

from environments.factory.base.renderer import RenderEntity
from environments.utility_classes import ObservationProperties


class Constants(BaseConstants):
    DIRT = 'Dirt'


class Actions(BaseActions):
    CLEAN_UP = 'do_cleanup_action'


class RewardsDirt(NamedTuple):
    CLEAN_UP_VALID: float          = 0.5
    CLEAN_UP_FAIL: float           = -0.1
    CLEAN_UP_LAST_PIECE: float     = 4.5


class DirtProperties(NamedTuple):
    initial_dirt_ratio: float = 0.3         # On INIT, on max how many tiles does the dirt spawn in percent.
    initial_dirt_spawn_r_var: float = 0.05   # How much does the dirt spawn amount vary?
    clean_amount: float = 1                 # How much does the robot clean with one actions.
    max_spawn_ratio: float = 0.20           # On max how many tiles does the dirt spawn in percent.
    max_spawn_amount: float = 0.3           # How much dirt does spawn per tile at max.
    spawn_frequency: int = 0                # Spawn Frequency in Steps.
    max_local_amount: int = 2               # Max dirt amount per tile.
    max_global_amount: int = 20             # Max dirt amount in the whole environment.
    dirt_smear_amount: float = 0.2          # Agents smear dirt, when not cleaning up in place.
    done_when_clean: bool = True


class Dirt(Entity):

    @property
    def amount(self):
        return self._amount

    @property
    def encoding(self):
        # Edit this if you want items to be drawn in the ops differntly
        return self._amount

    def __init__(self, *args, amount=None, **kwargs):
        super(Dirt, self).__init__(*args, **kwargs)
        self._amount = amount

    def set_new_amount(self, amount):
        self._amount = amount
        self._register.notify_change_to_value(self)

    def summarize_state(self, **kwargs):
        state_dict = super().summarize_state(**kwargs)
        state_dict.update(amount=float(self.amount))
        return state_dict


class DirtRegister(EntityRegister):

    _accepted_objects = Dirt

    @property
    def amount(self):
        return sum([dirt.amount for dirt in self])

    @property
    def dirt_properties(self):
        return self._dirt_properties

    def __init__(self, dirt_properties, *args):
        super(DirtRegister, self).__init__(*args)
        self._dirt_properties: DirtProperties = dirt_properties

    def spawn_dirt(self, then_dirty_tiles) -> bool:
        if isinstance(then_dirty_tiles, Floor):
            then_dirty_tiles = [then_dirty_tiles]
        for tile in then_dirty_tiles:
            if not self.amount > self.dirt_properties.max_global_amount:
                dirt = self.by_pos(tile.pos)
                if dirt is None:
                    dirt = Dirt(tile, self, amount=self.dirt_properties.max_spawn_amount)
                    self.register_item(dirt)
                else:
                    new_value = dirt.amount + self.dirt_properties.max_spawn_amount
                    dirt.set_new_amount(min(new_value, self.dirt_properties.max_local_amount))
            else:
                return c.NOT_VALID
        return c.VALID

    def __repr__(self):
        s = super(DirtRegister, self).__repr__()
        return f'{s[:-1]}, {self.amount})'


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def entropy(x):
    return -(x * np.log(x + 1e-8)).sum()


c = Constants
a = Actions


# noinspection PyAttributeOutsideInit, PyAbstractClass
class DirtFactory(BaseFactory):

    @property
    def actions_hook(self) -> Union[Action, List[Action]]:
        super_actions = super().actions_hook
        super_actions.append(Action(str_ident=a.CLEAN_UP))
        return super_actions

    @property
    def entities_hook(self) -> Dict[(str, Entities)]:
        super_entities = super().entities_hook
        dirt_register = DirtRegister(self.dirt_prop, self._level_shape)
        super_entities.update(({c.DIRT: dirt_register}))
        return super_entities

    def __init__(self, *args,
                 dirt_prop: DirtProperties = DirtProperties(), rewards_dirt: RewardsDirt = RewardsDirt(),
                 env_seed=time.time_ns(), **kwargs):
        if isinstance(dirt_prop, dict):
            dirt_prop = DirtProperties(**dirt_prop)
        if isinstance(rewards_dirt, dict):
            rewards_dirt = RewardsDirt(**rewards_dirt)
        self.dirt_prop = dirt_prop
        self.rewards_dirt = rewards_dirt
        self._dirt_rng = np.random.default_rng(env_seed)
        self._dirt: DirtRegister
        kwargs.update(env_seed=env_seed)
        # TODO: Reset ---> document this
        super().__init__(*args, **kwargs)

    def render_assets_hook(self, mode='human'):
        additional_assets = super().render_assets_hook()
        dirt = [RenderEntity('dirt', dirt.tile.pos, min(0.15 + dirt.amount, 1.5), 'scale')
                for dirt in self[c.DIRT]]
        additional_assets.extend(dirt)
        return additional_assets

    def do_cleanup_action(self, agent: Agent) -> (dict, dict):
        if dirt := self[c.DIRT].by_pos(agent.pos):
            new_dirt_amount = dirt.amount - self.dirt_prop.clean_amount

            if new_dirt_amount <= 0:
                self[c.DIRT].delete_env_object(dirt)
            else:
                dirt.set_new_amount(max(new_dirt_amount, c.FREE_CELL.value))
            valid = c.VALID
            self.print(f'{agent.name} did just clean up some dirt at {agent.pos}.')
            info_dict = {f'{agent.name}_{a.CLEAN_UP}_VALID': 1, 'cleanup_valid': 1}
            reward = self.rewards_dirt.CLEAN_UP_VALID
        else:
            valid = c.NOT_VALID
            self.print(f'{agent.name} just tried to clean up some dirt at {agent.pos}, but failed.')
            info_dict = {f'{agent.name}_{a.CLEAN_UP}_FAIL': 1, 'cleanup_fail': 1}
            reward = self.rewards_dirt.CLEAN_UP_FAIL

        if valid and self.dirt_prop.done_when_clean and (len(self[c.DIRT]) == 0):
            reward += self.rewards_dirt.CLEAN_UP_LAST_PIECE
            self.print(f'{agent.name} picked up the last piece of dirt!')
            info_dict = {f'{agent.name}_{a.CLEAN_UP}_LAST_PIECE': 1}
        return valid, dict(value=reward, reason=a.CLEAN_UP, info=info_dict)

    def trigger_dirt_spawn(self, initial_spawn=False):
        dirt_rng = self._dirt_rng
        free_for_dirt = [x for x in self[c.FLOOR]
                         if len(x.guests) == 0 or (len(x.guests) == 1 and isinstance(next(y for y in x.guests), Dirt))
                         ]
        self._dirt_rng.shuffle(free_for_dirt)
        if initial_spawn:
            var = self.dirt_prop.initial_dirt_spawn_r_var
            new_spawn = self.dirt_prop.initial_dirt_ratio + dirt_rng.uniform(-var, var)
        else:
            new_spawn = dirt_rng.uniform(0, self.dirt_prop.max_spawn_ratio)
        n_dirt_tiles = max(0, int(new_spawn * len(free_for_dirt)))
        self[c.DIRT].spawn_dirt(free_for_dirt[:n_dirt_tiles])

    def step_hook(self) -> (List[dict], dict):
        super_reward_info = super().step_hook()
        # if smear_amount := self.dirt_prop.dirt_smear_amount:
        #     for agent in self[c.AGENT]:
        #         if agent.temp_valid and agent.last_pos != c.NO_POS:
        #             if self._actions.is_moving_action(agent.temp_action):
        #                 if old_pos_dirt := self[c.DIRT].by_pos(agent.last_pos):
        #                     if smeared_dirt := round(old_pos_dirt.amount * smear_amount, 2):
        #                         old_pos_dirt.set_new_amount(max(0, old_pos_dirt.amount-smeared_dirt))
        #                         if new_pos_dirt := self[c.DIRT].by_pos(agent.pos):
        #                             new_pos_dirt.set_new_amount(max(0, new_pos_dirt.amount + smeared_dirt))
        #                         else:
        #                             if self[c.DIRT].spawn_dirt(agent.tile):
        #                                 new_pos_dirt = self[c.DIRT].by_pos(agent.pos)
        #                                 new_pos_dirt.set_new_amount(max(0, new_pos_dirt.amount + smeared_dirt))
        if self._next_dirt_spawn < 0:
            pass  # No Dirt Spawn
        elif not self._next_dirt_spawn:
            self.trigger_dirt_spawn()
            self._next_dirt_spawn = self.dirt_prop.spawn_frequency
        else:
            self._next_dirt_spawn -= 1
        return super_reward_info

    def do_additional_actions(self, agent: Agent, action: Action) -> (dict, dict):
        action_result = super().do_additional_actions(agent, action)
        if action_result is None:
            if action == a.CLEAN_UP:
                return self.do_cleanup_action(agent)
            else:
                return None
        else:
            return action_result

    def reset_hook(self) -> None:
        super().reset_hook()
        self.trigger_dirt_spawn(initial_spawn=True)
        self._next_dirt_spawn = self.dirt_prop.spawn_frequency if self.dirt_prop.spawn_frequency else -1

    def check_additional_done(self) -> (bool, dict):
        super_done, super_dict = super().check_additional_done()
        if self.dirt_prop.done_when_clean:
            if all_cleaned := len(self[c.DIRT]) == 0:
                super_dict.update(ALL_CLEAN_DONE=all_cleaned)
                return all_cleaned, super_dict
        return super_done, super_dict

    def observations_hook(self) -> Dict[str, np.typing.ArrayLike]:
        additional_observations = super().observations_hook()
        additional_observations.update({c.DIRT: self[c.DIRT].as_array()})
        return additional_observations

    def gather_additional_info(self, agent: Agent) -> dict:
        event_reward_dict = super().per_agent_reward_hook(agent)
        info_dict = dict()

        dirt = [dirt.amount for dirt in self[c.DIRT]]
        current_dirt_amount = sum(dirt)
        dirty_tile_count = len(dirt)

        # if dirty_tile_count:
        #    dirt_distribution_score = entropy(softmax(np.asarray(dirt)) / dirty_tile_count)
        # else:
        #    dirt_distribution_score = 0

        info_dict.update(dirt_amount=current_dirt_amount)
        info_dict.update(dirty_tile_count=dirty_tile_count)

        event_reward_dict.update({'info': info_dict})
        return event_reward_dict


if __name__ == '__main__':
    from environments.utility_classes import AgentRenderOptions as aro
    render = True

    dirt_props = DirtProperties(
        initial_dirt_ratio=0.35,
        initial_dirt_spawn_r_var=0.1,
        clean_amount=0.34,
        max_spawn_amount=0.1,
        max_global_amount=20,
        max_local_amount=1,
        spawn_frequency=0,
        max_spawn_ratio=0.05,
        dirt_smear_amount=0.0
    )

    obs_props = ObservationProperties(render_agents=aro.COMBINED, omit_agent_self=True,
                                      pomdp_r=2, additional_agent_placeholder=None, cast_shadows=True,
                                      indicate_door_area=False)

    move_props = {'allow_square_movement': True,
                  'allow_diagonal_movement': False,
                  'allow_no_op': False}
    import time
    global_timings = []
    for i in range(10):

        factory = DirtFactory(n_agents=10, done_at_collision=False,
                              level_name='rooms', max_steps=1000,
                              doors_have_area=False,
                              obs_prop=obs_props, parse_doors=True,
                              verbose=True,
                              mv_prop=move_props, dirt_prop=dirt_props,
                              # inject_agents=[TSPDirtAgent],
                              )

        factory.save_params(Path('rewards_param'))

        # noinspection DuplicatedCode
        n_actions = factory.action_space.n - 1
        _ = factory.observation_space
        obs_space = factory.observation_space
        obs_space_named = factory.named_observation_space
        action_space_named = factory.named_action_space
        times = []
        for epoch in range(10):
            start_time = time.time()
            random_actions = [[random.randint(0, n_actions) for _
                               in range(factory.n_agents)] for _
                              in range(factory.max_steps+1)]
            env_state = factory.reset()
            if render:
                factory.render()
            # tsp_agent = factory.get_injected_agents()[0]

            rwrd = 0
            for agent_i_action in random_actions:
                # agent_i_action = tsp_agent.predict()
                env_state, step_rwrd, done_bool, info_obj = factory.step(agent_i_action)
                rwrd += step_rwrd
                if render:
                    factory.render()
                if done_bool:
                    break
            times.append(time.time() - start_time)
            # print(f'Factory run {epoch} done, reward is:\n    {r}')
        print('Mean Time Taken: ', sum(times) / 10)
        global_timings.extend(times)
    print('Mean Time Taken: ', sum(global_timings) / len(global_timings))
    print('Median Time Taken: ', global_timings[len(global_timings)//2])

pass
