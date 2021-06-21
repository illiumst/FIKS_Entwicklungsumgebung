from typing import NamedTuple, Union, Iterable
from collections import namedtuple, deque
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.buffers import ReplayBuffer
import copy


class Experience(NamedTuple):
    observation:      np.ndarray
    next_observation: np.ndarray
    action:           np.ndarray
    reward:           Union[float, np.ndarray]
    done  :           Union[bool, np.ndarray]
    priority:         np.ndarray = 1


class BaseBuffer:
    def __init__(self, size: int):
        self.size = size
        self.experience = deque(maxlen=size)

    def __len__(self):
        return len(self.experience)

    def add(self, experience):
        self.experience.append(experience)

    def sample(self, k, cer=4):
        sample = random.choices(self.experience, k=k-cer)
        for i in range(cer): sample += [self.experience[-i]]
        observations = torch.stack([torch.from_numpy(e.observation) for e in sample], 0).float()
        next_observations = torch.stack([torch.from_numpy(e.next_observation) for e in sample], 0).float()
        actions = torch.tensor([e.action for e in sample]).long()
        rewards = torch.tensor([e.reward for e in sample]).float().view(-1, 1)
        dones = torch.tensor([e.done for e in sample]).float().view(-1, 1)
        return Experience(observations, next_observations, actions, rewards, dones)


class BaseDQN(nn.Module):
    def __init__(self):
        super(BaseDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*5*5, 64),
            nn.ELU(),
            nn.Linear(64,  64),
            nn.ELU()
        )
        self.value_head         =  nn.Linear(64, 1)
        self.advantage_head     =  nn.Linear(64, 9)

    def act(self, x) -> np.ndarray:
        with torch.no_grad():
            action = self.forward(x).max(-1)[1].numpy()
        return action

    def forward(self, x):
        features = self.net(x)
        advantages = self.advantage_head(features)
        values = self.value_head(features)
        return values + (advantages - advantages.mean())

    def random_action(self):
        return random.randrange(0, 5)


def soft_update(local_model, target_model, tau):
    # taken from https://github.com/BY571/Munchausen-RL/blob/master/M-DQN.ipynb
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.-tau)*target_param.data)


class BaseQlearner:
    def __init__(self, q_net, target_q_net, env, buffer, target_update, eps_end, n_agents=1,
                 gamma=0.99, train_every_n_steps=4, n_grad_steps=1, tau=1.0, max_grad_norm=10,
                 exploration_fraction=0.2, batch_size=64, lr=1e-4, reg_weight=0.0):
        self.q_net = q_net
        self.target_q_net = target_q_net
        #self.q_net.apply(self.weights_init)
        self.target_q_net.eval()
        soft_update(self.q_net, self.target_q_net, tau=1.0)
        self.env = env
        self.buffer = buffer
        self.target_update = target_update
        self.eps = 1.
        self.eps_end = eps_end
        self.exploration_fraction = exploration_fraction
        self.batch_size = batch_size
        self.gamma = gamma
        self.train_every_n_steps = train_every_n_steps
        self.n_grad_steps = n_grad_steps
        self.lr = lr
        self.tau = tau
        self.reg_weight = reg_weight
        self.n_agents = n_agents
        self.device = 'cpu'
        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=self.lr)
        self.max_grad_norm = max_grad_norm
        self.running_reward = deque(maxlen=5)
        self.running_loss = deque(maxlen=5)
        self._n_updates = 0

    def to(self, device):
        self.device = device
        return self

    @staticmethod
    def weights_init(module, activation='leaky_relu'):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(module.weight, gain=torch.nn.init.calculate_gain(activation))
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def anneal_eps(self, step, n_steps):
        fraction = min(float(step) / int(self.exploration_fraction*n_steps), 1.0)
        self.eps = 1 + fraction * (self.eps_end - 1)

    def get_action(self, obs) -> Union[int, np.ndarray]:
        o = torch.from_numpy(obs).unsqueeze(0) if self.n_agents <= 1 else torch.from_numpy(obs)
        if np.random.rand() > self.eps:
            action = self.q_net.act(o.float())
        else:
            action = np.array([self.env.action_space.sample() for _ in range(self.n_agents)])
        return action

    def learn(self, n_steps):
        step = 0
        while step < n_steps:
            obs, done = self.env.reset(), False
            total_reward = 0
            while not done:

                action = self.get_action(obs)

                next_obs, reward, done, info = self.env.step(action if not len(action) == 1 else action[0])

                experience = Experience(observation=obs, next_observation=next_obs, action=action, reward=reward, done=done)  # do we really need to copy?
                self.buffer.add(experience)
                # end of step routine
                obs = next_obs
                step += 1
                total_reward += reward
                self.anneal_eps(step, n_steps)

                if step % self.train_every_n_steps == 0:
                    self.train()
                    self._n_updates += 1
                if step % self.target_update == 0:
                    print('UPDATE')
                    soft_update(self.q_net, self.target_q_net, tau=self.tau)

            self.running_reward.append(total_reward)
            if step % 10 == 0:
                print(f'Step: {step} ({(step/n_steps)*100:.2f}%)\tRunning reward: {sum(list(self.running_reward))/len(self.running_reward):.2f}\t'
                      f' eps: {self.eps:.4f}\tRunning loss: {sum(list(self.running_loss))/len(self.running_loss):.4f}\tUpdates:{self._n_updates}')

    def _training_routine(self, obs, next_obs, action, reward):
        current_q_values = self.q_net(obs)
        current_q_values = torch.gather(current_q_values, dim=-1, index=action)
        next_q_values_raw = self.target_q_net(next_obs).max(dim=-1)[0].reshape(-1, 1).detach()
        return current_q_values, next_q_values_raw


    def train(self):
        if len(self.buffer) < self.batch_size: return
        for _ in range(self.n_grad_steps):

            experience = self.buffer.sample(self.batch_size, cer=self.train_every_n_steps)

            if self.n_agents <= 1:
                pred_q, target_q_raw = self._training_routine(experience.observation,
                                                                      experience.next_observation,
                                                                      experience.action,
                                                                      experience.reward)
            else:
                pred_q, target_q_raw, reward = [torch.zeros((self.batch_size, 1))]*3
                for agent_i in range(self.n_agents):
                    q_values, next_q_values_raw = self._training_routine(experience.observation[:, agent_i],
                                                                                   experience.next_observation[:, agent_i],
                                                                                   experience.action[:, agent_i].unsqueeze(-1),
                                                                                   experience.reward)
                    pred_q += q_values
                    target_q_raw += next_q_values_raw
            target_q = experience.reward  + (1 - experience.done) * self.gamma * target_q_raw
            loss = torch.mean(self.reg_weight * pred_q + torch.pow(pred_q - target_q, 2))

            # log loss
            self.running_loss.append(loss.item())
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.optimizer.step()


class MDQN(BaseQlearner):
    def __init__(self, *args, temperature=0.03, alpha=0.9, clip_l0=-1.0, **kwargs):
        super(MDQN, self).__init__(*args, **kwargs)
        assert self.n_agents == 1, 'M-DQN currently only supports single agent training'
        self.temperature = temperature
        self.alpha = alpha
        self.clip0 = clip_l0

    def tau_ln_pi(self, qs):
        # Custom log-sum-exp trick from page 18 to compute the e log-policy terms
        v_k = qs.max(-1)[0].unsqueeze(-1)
        advantage = qs - v_k
        logsum = torch.logsumexp(advantage / self.temperature, -1).unsqueeze(-1)
        tau_ln_pi = advantage - self.temperature * logsum
        return tau_ln_pi

    def train(self):
        if len(self.buffer) < self.batch_size: return
        for _ in range(self.n_grad_steps):

            experience = self.buffer.sample(self.batch_size, cer=self.train_every_n_steps)

            q_target_next = self.target_q_net(experience.next_observation).detach()
            tau_log_pi_next = self.tau_ln_pi(q_target_next)

            q_k_targets = self.target_q_net(experience.observation).detach()
            log_pi = self.tau_ln_pi(q_k_targets)

            pi_target = F.softmax(q_target_next / self.temperature, dim=-1)
            q_target = (self.gamma * (pi_target * (q_target_next - tau_log_pi_next) * (1 - experience.done)).sum(-1)).unsqueeze(-1)

            munchausen_addon = log_pi.gather(-1, experience.action)

            munchausen_reward = (experience.reward + self.alpha * torch.clamp(munchausen_addon, min=self.clip0, max=0))

            # Compute Q targets for current states
            m_q_target = munchausen_reward + q_target

            # Get expected Q values from local model
            q_k = self.q_net(experience.observation)
            pred_q = q_k.gather(-1, experience.action)

            # Compute loss
            loss = torch.mean(self.reg_weight * pred_q + torch.pow(pred_q - m_q_target, 2))

            # log loss
            self.running_loss.append(loss.item())
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.optimizer.step()



if __name__ == '__main__':
    from environments.factory.simple_factory import SimpleFactory, DirtProperties, MovementProperties
    from algorithms.reg_dqn import RegDQN
    from stable_baselines3.common.vec_env import DummyVecEnv

    N_AGENTS = 1

    dirt_props = DirtProperties(clean_amount=3, gain_amount=0.2, max_global_amount=30,
                                max_local_amount=5, spawn_frequency=1, max_spawn_ratio=0.05)
    move_props = MovementProperties(allow_diagonal_movement=True,
                                    allow_square_movement=True,
                                    allow_no_op=False)
    env = SimpleFactory(dirt_properties=dirt_props, movement_properties=move_props, n_agents=N_AGENTS, pomdp_radius=2,  max_steps=400, omit_agent_slice_in_obs=False, combin_agent_slices_in_obs=True)
    #env = DummyVecEnv([lambda: env])
    from stable_baselines3.dqn import DQN

    #dqn = RegDQN('MlpPolicy', env, verbose=True, buffer_size = 40000, learning_starts = 0, batch_size = 64,learning_rate=0.0008,
    #             target_update_interval = 3500, exploration_fraction = 0.25, exploration_final_eps = 0.05,
    #             train_freq=4, gradient_steps=1, reg_weight=0.05, seed=69)
    #dqn.learn(100000)


    dqn, target_dqn = BaseDQN(), BaseDQN()
    learner = MDQN(dqn, target_dqn, env, BaseBuffer(40000), target_update=3500, lr=0.0008, gamma=0.99, n_agents=N_AGENTS, tau=0.95, max_grad_norm=10,
                   train_every_n_steps=4, eps_end=0.025, n_grad_steps=1, reg_weight=0.1, exploration_fraction=0.25, batch_size=64)
    learner.learn(100000)
