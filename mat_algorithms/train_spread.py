from pettingzoo.mpe import simple_spread_v3
import torch
import numpy as np
from torch.distributions import Categorical
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from mat.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy
from mat.utils.shared_buffer import SharedReplayBuffer

from gymnasium.spaces import Box
from argparse import Namespace
import os


env = simple_spread_v3.parallel_env(N=2, max_cycles=50, render_mode=None)
env.reset()

all_args_dict = {
    "episode_length": 50,
    "n_rollout_threads": 1,
    "hidden_size": 128,
    "recurrent_N": 1,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "use_gae": True,
    "use_popart": False,
    "use_valuenorm": True,
    "use_proper_time_limits": False,
    "algorithm_name": "mat",
    "clip_param": 0.1,
    "ppo_epoch": 20,
    "num_mini_batch": 1,
    "data_chunk_length": 10,
    "value_loss_coef": 1,
    "entropy_coef": 0.1,
    "max_grad_norm": 1.0,
    "huber_delta": 10.0,
    "use_recurrent_policy": False,
    "use_naive_recurrent_policy": False,
    "use_max_grad_norm": True,
    "use_clipped_value_loss": True,
    "use_huber_loss": True,
    "use_value_active_masks": True,
    "use_policy_active_masks": True,
    "dec_actor": False,
    "lr": 0.0005,
    "opti_eps": 1e-05,
    "weight_decay": 0,
    "n_block": 3,
    "n_embd": 512,
    "n_head": 8,
    "encode_state": False,
    "env_name": "MPE",
    "share_actor": False,
    "use_centralized_V": True
}

all_args = Namespace(**all_args_dict)

def _t2n(x):
    return x.detach().cpu().numpy()

class MARLTrainer:
    def __init__(self, env, all_args):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_agents = len(self.env.agents)
        self.state_dim = sum(space.shape[0] for space in self.env.observation_spaces.values())
        self.action_dim = self.env.action_space(self.env.agents[0]).n

        shared_obs = self.create_shared_observation(self.env.observation_spaces.values())
        self.env.shared_observations = shared_obs
        
        self.all_args = all_args

        self.policy = Policy(
            self.all_args, self.env.observation_space(self.env.agents[0]), 
            shared_obs,
            env.action_space(self.env.agents[0]), self.n_agents, 
            device=self.device
        )
        
        self.trainer = TrainAlgo(self.all_args, self.policy, self.n_agents, device=self.device)
        self.buffer = SharedReplayBuffer(
            self.all_args, self.n_agents, env.observation_space(env.agents[0]), 
            shared_obs, 
            env.action_space(env.agents[0]), self.all_args.env_name
        )
        
        self.current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.writer = SummaryWriter(log_dir=f'spread_logs/run_{self.current_time}')
        self.save_dir = "spread_models/"
        os.makedirs(self.save_dir, exist_ok=True)

    @staticmethod
    def create_shared_observation(observation_spaces):
        dims = sum(space.shape[0] for space in observation_spaces)
        return Box(low=-np.inf, high=np.inf, shape=(dims,), dtype=np.float32)

    def concat_observation(self, obs):
        share_obs = obs.reshape(self.all_args.n_rollout_threads, -1)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.n_agents, axis=1)
        return share_obs

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step])
        )

        values = np.array(np.split(_t2n(value), self.all_args.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.all_args.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.all_args.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.all_args.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.all_args.n_rollout_threads))

        if self.env.action_space(self.env.agents[0]).__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.env.action_space(self.env.agents[0]).n)[actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        shared_obs, obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rewards = np.expand_dims(np.expand_dims(rewards, axis=0), axis=-1)
        dones = np.array([dones])
        
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.all_args.recurrent_N, self.all_args.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.all_args.n_rollout_threads, self.n_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        self.buffer.insert(shared_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1])
        )

        next_values = np.array(np.split(_t2n(next_values), self.all_args.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def save(self, episode=0):
        self.policy.save(self.save_dir, episode)

    def train(self):
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def train_agents(self, episodes=1000, save_interval=100):
        for episode in range(episodes):
            observations, infos = self.env.reset()

            obs_tensor = torch.tensor([observations[env.agents[0]]], dtype=torch.float32, device=self.device).unsqueeze(0)
            state_tensor = torch.zeros((1, 1, self.state_dim), dtype=torch.float32, device=self.device)
            masks = np.ones((self.all_args.n_rollout_threads, self.n_agents, 1), dtype=np.float32)

            # shared_obs = self.concat_observation(np.array(list(observations.values())))

            # self.buffer.share_obs[0] = shared_obs.copy()
            # self.buffer.obs[0] = np.array(list(observations.values())).copy()

            total_rewards = np.zeros(self.n_agents)
            step = 0

            while self.env.agents:
                values_return, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                actions_dict = {agent: actions[0, i].item() for i, agent in enumerate(self.env.agents)}

                observations, rewards, terminations, truncations, infos = self.env.step(actions_dict)

                shared_obs = self.concat_observation(np.array(list(observations.values())))
                total_rewards += np.array(list(rewards.values()))
                dones = np.array([terminations[k] or truncations[k] for k in terminations])

                data = shared_obs, np.array(list(observations.values())), np.array(list(rewards.values())), dones, infos, values_return, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)

                step += 1

            if episode % 500 == 0:
                print(f"Episode {episode}: Actions taken: {actions_dict}")

            if episode % 500 == 0:
                print(f"Episode {episode}: Total Reward: {np.sum(total_rewards)}")

            if episode % 500 == 0:
                sampled_actions = self.policy.act(
                    shared_obs, np.array(list(observations.values())), state_tensor.cpu(), masks, deterministic=False
                )
                print(f"Sampled Actions at Episode {episode}: {sampled_actions}")
        
            self.compute()
            train_info = self.train()

            buffer_rewards = np.mean(self.buffer.rewards) * self.all_args.episode_length
            # avg_reward = total_rewards.mean()
            for i, reward in enumerate(total_rewards):
                self.writer.add_scalar(f'Reward/Agent_{i}', reward, episode)
            self.writer.add_scalar('Reward/average_episode_reward', buffer_rewards, episode)

            if episode % save_interval == 0 or episode == episodes - 1:
                self.save(episode)

            
            # print(f"Episode {episode + 1}/{episodes}, Rewards: {total_rewards}, Avg Reward: {avg_reward:.2f}")
            agent_rewards_str = ', '.join([f"Agent_{i}: {reward:.2f}" for i, reward in enumerate(total_rewards)])
            print(f"Episode {episode + 1}/{episodes}, {agent_rewards_str}, Average Episode Reward: {buffer_rewards:.2f}")

        # self.writer.close()

    def evaluate(self, episodes=10, model_path=None):
        if model_path:
            self.policy.restore(model_path)

        for episode in range(episodes):
            observations, infos = self.env.reset()

            obs_tensor = torch.tensor([observations[env.agents[0]]], dtype=torch.float32, device=self.device).unsqueeze(0)
            state_tensor = torch.zeros((1, 1, self.state_dim), dtype=torch.float32, device=self.device)
            masks = np.ones((self.all_args.n_rollout_threads, self.n_agents, 1), dtype=np.float32)

            total_rewards = np.zeros(self.n_agents)

            while self.env.agents:
                shared_obs = self.concat_observation(list(observations.values()))
                shared_obs = np.tile(shared_obs, (self.n_agents, 1))

                self.trainer.prep_rollout()
                actions, _ = self.trainer.policy.act(
                    shared_obs, np.array(list(observations.values())), state_tensor.cpu(), masks, deterministic=True
                )

                actions_dict = {agent: actions.cpu().numpy()[i, 0].item() for i, agent in enumerate(self.env.agents)}
                next_observations, rewards, terminations, truncations, infos = self.env.step(actions_dict)

                total_rewards += np.array(list(rewards.values()))
                self.env.render()
                observations = next_observations

            print(f"Eval Episode {episode + 1}/{episodes}, Rewards: {total_rewards}")


trainer = MARLTrainer(env, all_args)
# train
trainer.train_agents(episodes=50000)

# evaluate
# trainer.evaluate(episodes=10, model_path='transformer_Feb18_19-24-44.pt')