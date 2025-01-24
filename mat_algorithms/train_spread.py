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

# Function to create shared observation
def create_shared_observation(observation_spaces):
    dims = sum(space.shape[0] for space in observation_spaces)
    return Box(low=-np.inf, high=np.inf, shape=(dims,), dtype=np.float32)

# Concat existing agent observations
def concat_observation(observations):
    concat_obs = np.concatenate(observations)
    
    return concat_obs


env = simple_spread_v3.parallel_env(N=3, max_cycles=50, render_mode=None)
env.reset()

shared_obs = create_shared_observation(env.observation_spaces.values())
env.shared_observations = shared_obs


# Model hyperparameters
n_agents = len(env.agents)
obs_dim = env.observation_space(env.agents[0]).shape[0]
action_dim = env.action_space(env.agents[0]).n
state_dim = sum(space.shape[0] for space in env.observation_spaces.values()) #was 37 for simple_tag 
n_block = 3
n_embd = 512
n_head = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_args_dict = {
    "episode_length": 50,
    "n_rollout_threads": 1,
    "hidden_size": 64,
    "recurrent_N": 1,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "use_gae": True,
    "use_popart": False,
    "use_valuenorm": True,
    "use_proper_time_limits": False,
    "algorithm_name": "mat",
    "clip_param": 0.2,
    "ppo_epoch": 10,
    "num_mini_batch": 1,
    "data_chunk_length": 10,
    "value_loss_coef": 1,
    "entropy_coef": 0.01,
    "max_grad_norm": 10.0,
    "huber_delta": 10.0,
    "use_recurrent_policy": False,
    "use_naive_recurrent_policy": False,
    "use_max_grad_norm": True,
    "use_clipped_value_loss": True,
    "use_huber_loss": True,
    "use_value_active_masks": True,
    "use_policy_active_masks": True,
    "dec_actor": False,
    "lr": 0.0001,
    "opti_eps": 1e-05,
    "weight_decay": 0,
    "n_block": 1,
    "n_embd": 64,
    "n_head": 1,
    "encode_state": False,
    "env_name": "MPE",
    "share_actor": False,
    "use_centralized_V": True
}

all_args = Namespace(**all_args_dict)

policy = Policy(all_args, env.observation_space(env.agents[0]), env.shared_observations, env.action_space(env.agents[0]), n_agents, device=device)
trainer = TrainAlgo(all_args, policy, n_agents, device=device)
buffer = SharedReplayBuffer(all_args, n_agents,  env.observation_space(env.agents[0]), env.shared_observations, env.action_space(env.agents[0]), all_args.env_name)


# TensorBoard writer setup
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
writer = SummaryWriter(log_dir=f'spread_logs/run_{current_time}')


# Directory to save models
save_dir = f"spread_models/"
os.makedirs(save_dir, exist_ok=True)

# Directory to load model
model_dir = f"models/transformer_Jan16_20-43-20.pt"

# COPIED functions

def _t2n(x):
    return x.detach().cpu().numpy()

@torch.no_grad()
def collect(step):
    trainer.prep_rollout()
    value, action, action_log_prob, rnn_states, rnn_states_critic \
        = trainer.policy.get_actions(np.concatenate(buffer.share_obs[step]),
                        np.concatenate(buffer.obs[step]),
                        np.concatenate(buffer.rnn_states[step]),
                        np.concatenate(buffer.rnn_states_critic[step]),
                        np.concatenate(buffer.masks[step]))
    # [self.envs, agents, dim]
    values = np.array(np.split(_t2n(value), all_args.n_rollout_threads))
    actions = np.array(np.split(_t2n(action), all_args.n_rollout_threads))
    action_log_probs = np.array(np.split(_t2n(action_log_prob), all_args.n_rollout_threads))
    rnn_states = np.array(np.split(_t2n(rnn_states), all_args.n_rollout_threads))
    rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), all_args.n_rollout_threads))
    # rearrange action
    if env.action_space(env.agents[0]).__class__.__name__ == 'Discrete':
        actions_env = np.squeeze(np.eye(env.action_space(env.agents[0]).n)[actions], 2)
    else:
        raise NotImplementedError

    return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

def insert(data):
    shared_obs, obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

    # Reshape rewards to match the expected shape (1, n_agents, 1)
    rewards = np.expand_dims(np.expand_dims(rewards, axis=0), axis=-1)

    # dones = []
    # dones.append(dones1)
    dones = np.array([dones])

    rnn_states[dones == True] = np.zeros(((dones == True).sum(), all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
    rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
    masks = np.ones((all_args.n_rollout_threads, n_agents, 1), dtype=np.float32)
    masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

    buffer.insert(shared_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

@torch.no_grad()
def compute():
    """Calculate returns for the collected data."""
    trainer.prep_rollout()
    next_values = trainer.policy.get_values(np.concatenate(buffer.share_obs[-1]),
                                                np.concatenate(buffer.obs[-1]),
                                                np.concatenate(buffer.rnn_states_critic[-1]),
                                                np.concatenate(buffer.masks[-1]))

    next_values = np.array(np.split(_t2n(next_values), all_args.n_rollout_threads))
    buffer.compute_returns(next_values, trainer.value_normalizer)

def save(episode=0):
    """Save policy's actor and critic networks."""
    policy.save(save_dir, episode)

def train():
    """Train adversary policies with data in buffer. """
    trainer.prep_training()
    train_infos = trainer.train(buffer)      
    buffer.after_update()
    return train_infos

# Training function
def train_agents(env, episodes=1000, gamma=0.99, save_interval=100):
    for episode in range(episodes):
        observations, infos = env.reset()

        obs_tensor = torch.tensor([observations[env.agents[0]]], dtype=torch.float32, device=device).unsqueeze(0)
        state_tensor = torch.zeros((1, 1, state_dim), dtype=torch.float32, device=device)

        total_rewards = np.zeros(n_agents)
        step = 0

        while env.agents:
            
            # Get action for the agents
            values_return, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = collect(step)

            # Step environment
            actions_dict = {agent: actions[0, i, 0] for i, agent in enumerate(env.agents)}
            next_observations, rewards, terminations, truncations, infos = env.step(actions_dict)

            shared_obs = concat_observation(list(next_observations.values()))

            total_rewards += np.array(list(rewards.values()))
            dones = {k: terminations[k] or truncations[k] for k in terminations}
            dones = np.array(list(dones.values()))

            data_ad = shared_obs, list(observations.values()), total_rewards, dones, infos, values_return, actions, action_log_probs, rnn_states, rnn_states_critic

            insert(data_ad)
            
            observations = next_observations
            step += 1

        compute()
        train_info = train()

        buffer_rewards = np.mean(buffer.rewards) * all_args.episode_length

        # Calculate average reward across agents
        avg_reward = total_rewards.mean()

        # Log rewards for each agent and the average reward
        for i, reward in enumerate(total_rewards):
            writer.add_scalar(f'Reward/Agent_{i}', reward, episode)
        writer.add_scalar('Reward/average_episode_reward', buffer_rewards, episode)

        # Save the model periodically
        if (episode % save_interval == 0 or episode == episodes - 1):
            save()

        # Print rewards for each agent and the average reward
        agent_rewards_str = ', '.join([f"Agent_{i}: {reward:.2f}" for i, reward in enumerate(total_rewards)])
        print(f"Episode {episode + 1}/{episodes}, {agent_rewards_str}, Average Episode Reward: {buffer_rewards:.2f}")


    # Close TensorBoard writer
    writer.close()


def evaluate(env, episodes=10):
    """Evaluate the trained adversary model in the environment."""
    # Load the latest trained model
    policy.restore(model_dir)

    total_rewards = []

    for episode in range(episodes):
        observations, infos = env.reset()

        obs_tensor = torch.tensor([observations[env.agents[0]]], dtype=torch.float32, device=device).unsqueeze(0)
        state_tensor = torch.zeros((1, 1, state_dim), dtype=torch.float32, device=device)
        masks = np.ones((all_args.n_rollout_threads, n_agents, 1), dtype=np.float32)

        total_reward = 0

        while env.agents:
            adversary = env.agents[0]
            good_agent = env.agents[1]

            shared_obs = concat_observation(list(observations.values()))

            # Get action for the adversary
            trainer.prep_rollout()
            action, _ = trainer.policy.act(shared_obs, observations[adversary], state_tensor, masks, deterministic=True)

            action = action.cpu().numpy()[0, 0]  # Select action from tensor

            # Manually set the good agent's action
            good_action = 0

            # Step the environment with both agents
            actions_dict = {adversary: action, good_agent: good_action}
            observations, rewards_dict, terminations, truncations, infos = env.step(actions_dict)

            # Render the environment
            env.render()

            total_reward += rewards_dict[adversary]

            # Prepare inputs for the next step
            if adversary in observations:
                obs_tensor = torch.tensor([observations[adversary]], dtype=torch.float32, device=device).unsqueeze(0)

        total_rewards.append(total_reward)
        print(f"Evaluation Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {episodes} Evaluation Episodes: {avg_reward:.2f}")

# Evaluate the trained adversary
# evaluate(env, episodes=10)

# Train the adversary
train_agents(env, episodes=50000)