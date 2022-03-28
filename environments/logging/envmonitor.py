import pickle
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import List, Dict, Union

from stable_baselines3.common.callbacks import BaseCallback

from environments.helpers import IGNORED_DF_COLUMNS

import pandas as pd

from plotting.compare_runs import plot_single_run


class EnvMonitor(BaseCallback):

    ext = 'png'

    def __init__(self, env, filepath: Union[str, PathLike] = None):
        super(EnvMonitor, self).__init__()
        self.unwrapped = env
        self._filepath = filepath
        self._monitor_df = pd.DataFrame()
        self._monitor_dicts = defaultdict(dict)

    def __getattr__(self, item):
        return getattr(self.unwrapped, item)

    def step(self, action):
        obs, reward, done, info = self.unwrapped.step(action)
        self._read_info(0, info)
        self._read_done(0, done)
        return obs, reward, done, info

    def reset(self):
        return self.unwrapped.reset()

    def _on_training_start(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass

    def _on_step(self, alt_infos: List[Dict] = None, alt_dones: List[bool] = None) -> bool:
        for env_idx, info in enumerate(self.locals.get('infos', [])):
            self._read_info(env_idx, info)

        for env_idx, done in list(
                enumerate(self.locals.get('dones', []))): # + list(enumerate(self.locals.get('done', []))):
            self._read_done(env_idx, done)
        return True

    def _read_info(self, env_idx, info: dict):
        self._monitor_dicts[env_idx][len(self._monitor_dicts[env_idx])] = {
            key: val for key, val in info.items() if
            key not in ['terminal_observation', 'episode'] and not key.startswith('rec_')}
        return

    def _read_done(self, env_idx, done):
        if done:
            env_monitor_df = pd.DataFrame.from_dict(self._monitor_dicts[env_idx], orient='index')
            self._monitor_dicts[env_idx] = dict()
            columns = [col for col in env_monitor_df.columns if col not in IGNORED_DF_COLUMNS]
            env_monitor_df = env_monitor_df.aggregate(
                {col: 'mean' if col.endswith('ount') else 'sum' for col in columns}
            )
            env_monitor_df['episode'] = len(self._monitor_df)
            self._monitor_df = self._monitor_df.append([env_monitor_df])
        else:
            pass
        return

    def save_run(self, filepath: Union[Path, str], auto_plotting_keys=None):
        filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True, parents=True)
        with filepath.open('wb') as f:
            pickle.dump(self._monitor_df.reset_index(), f, protocol=pickle.HIGHEST_PROTOCOL)
        if auto_plotting_keys:
            plot_single_run(filepath, column_keys=auto_plotting_keys)
