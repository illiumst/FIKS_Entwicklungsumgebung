import warnings
from pathlib import Path

import yaml
from natsort import natsorted
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from environments.factory.simple_factory import DirtProperties, SimpleFactory
from environments.factory.double_task_factory import ItemProperties, DoubleTaskFactory

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


if __name__ == '__main__':

    model_name = 'A2C_1630073286'
    run_id = 0
    out_path = Path(__file__).parent / 'debug_out'
    model_path = out_path / model_name

    with (model_path / f'env_{model_name}.yaml').open('r') as f:
        env_kwargs = yaml.load(f, Loader=yaml.FullLoader)
    if False:
        env_kwargs.update(dirt_properties=DirtProperties(clean_amount=1, gain_amount=0.1, max_global_amount=20,
                                                         max_local_amount=1, spawn_frequency=5, max_spawn_ratio=0.05,
                                                         dirt_smear_amount=0.5),
                          combin_agent_slices_in_obs=True, omit_agent_slice_in_obs=True)
    with SimpleFactory(**env_kwargs) as env:

        # Edit THIS:
        model_files = list(natsorted((model_path / f'{run_id}_{model_name}').rglob('model_*.zip')))
        this_model = model_files[0]

        model = PPO.load(this_model)
        evaluation_result = evaluate_policy(model, env, n_eval_episodes=100, deterministic=False, render=True)
        print(evaluation_result)
