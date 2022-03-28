import warnings
from pathlib import Path

import yaml

from stable_baselines3 import PPO

from environments.factory.factory_dirt import DirtProperties, DirtFactory, RewardsDirt
from environments.logging.envmonitor import EnvMonitor
from environments.logging.recorder import EnvRecorder
from environments.utility_classes import MovementProperties, ObservationProperties, AgentRenderOptions
from environments.factory.factory_dirt import Constants as c

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

if __name__ == '__main__':
    TRAIN_AGENT = True
    LOAD_AND_REPLAY = True
    record = True
    render = False

    study_root_path = Path(__file__).parent.parent / 'experiment_out'

    parameter_path = Path(__file__).parent.parent / 'environments' / 'factory' / 'levels' / 'parameters' / 'DirtyFactory-v0.yaml'

    save_path = study_root_path / f'model.zip'

    # Output folder

    study_root_path.mkdir(parents=True, exist_ok=True)

    train_steps = 2*1e5
    frames_to_stack = 0

    u = dict(
        show_global_position_info=True,
        pomdp_r=3,
        cast_shadows=True,
        allow_diagonal_movement=False,
        parse_doors=True,
        doors_have_area=False,
        done_at_collision=True
    )
    obs_props = ObservationProperties(render_agents=AgentRenderOptions.SEPERATE,
                                      additional_agent_placeholder=None,
                                      omit_agent_self=True,
                                      frames_to_stack=frames_to_stack,
                                      pomdp_r=u['pomdp_r'], cast_shadows=u['cast_shadows'],
                                      show_global_position_info=u['show_global_position_info'])
    move_props = MovementProperties(allow_diagonal_movement=u['allow_diagonal_movement'],
                                    allow_square_movement=True,
                                    allow_no_op=False)
    dirt_props = DirtProperties(initial_dirt_ratio=0.35, initial_dirt_spawn_r_var=0.1,
                                clean_amount=0.34,
                                max_spawn_amount=0.1, max_global_amount=20,
                                max_local_amount=1, spawn_frequency=0, max_spawn_ratio=0.05,
                                dirt_smear_amount=0.0)
    rewards_dirt = RewardsDirt(CLEAN_UP_FAIL=-0.5, CLEAN_UP_VALID=1, CLEAN_UP_LAST_PIECE=5)
    factory_kwargs = dict(n_agents=1, max_steps=500, parse_doors=u['parse_doors'],
                          level_name='rooms', doors_have_area=u['doors_have_area'],
                          verbose=True,
                          mv_prop=move_props,
                          obs_prop=obs_props,
                          rewards_dirt=rewards_dirt,
                          done_at_collision=u['done_at_collision']
                          )

    # with (parameter_path).open('r') as f:
    #     factory_kwargs = yaml.load(f, Loader=yaml.FullLoader)
    #     factory_kwargs.update(n_agents=1, done_at_collision=False, verbose=True)

    if TRAIN_AGENT:
        env = DirtFactory(**factory_kwargs)
        callbacks = EnvMonitor(env)
        obs_shape = env.observation_space.shape

        model = PPO("MlpPolicy", env, verbose=1, device='cpu')

        model.learn(total_timesteps=train_steps, callback=callbacks)

        callbacks.save_run(study_root_path / 'monitor.pick', auto_plotting_keys=['step_reward', 'collision'] + ['cleanup_valid', 'cleanup_fail']) # + env_plot_keys)


        model.save(save_path)

    if LOAD_AND_REPLAY:
        with DirtFactory(**factory_kwargs) as env:
            env = EnvMonitor(env)
            env = EnvRecorder(env) if record else env
            obs_shape = env.observation_space.shape
            model = PPO.load(save_path)
            # Evaluation Loop for i in range(n Episodes)
            for episode in range(10):
                env_state = env.reset()
                rew, done_bool = 0, False
                while not done_bool:
                    actions = model.predict(env_state, deterministic=True)[0]
                    env_state, step_r, done_bool, info_obj = env.step(actions)

                    rew += step_r

                    if render:
                        env.render()

                    try:
                        door = next(x for x in env.unwrapped.unwrapped.unwrapped[c.DOORS] if x.is_open)
                        print('openDoor found')
                    except StopIteration:
                        pass

                    if done_bool:
                        break
                print(
                    f'Factory run {episode} done, steps taken {env.unwrapped.unwrapped.unwrapped._steps}, reward is:\n    {rew}')

            env.save_records(study_root_path / 'reload_recorder.pick', save_occupation_map=False)
            #env.save_run(study_root_path / 'reload_monitor.pick',
            #             auto_plotting_keys=['step_reward', 'cleanup_valid', 'cleanup_fail'])