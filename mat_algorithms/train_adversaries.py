from pettingzoo.mpe import simple_tag_v3
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

# Initialize the environment with only 1 adversary and 1 good agent
def custom_env():
    env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=1, num_obstacles=1, max_cycles=100, render_mode=None)
    env.reset()

    # Keep only one adversary and one good agent
    adversary = [agent for agent in env.agents if "adversary" in agent][0]
    good_agent = [agent for agent in env.agents if "agent" in agent][0]
    env.agents = [adversary, good_agent]

    shared_obs = create_shared_observation(env.observation_spaces.values())
    env.shared_observations = shared_obs

    return env

def create_shared_observation(observation_spaces):
    dims = 0

    for observation_space in observation_spaces:
        dims += observation_space.shape[0]
    
    return Box(low=-np.inf, high=np.inf, shape=(dims,), dtype=np.float32)

# Concat existing agent observations
def concat_observation(observations):
    concat_obs = np.concatenate(observations)
    
    return concat_obs


env = custom_env()
current_time = datetime.now().strftime("%b%d_%H-%M-%S")

# TensorBoard writer setup
writer = SummaryWriter(log_dir=f'logs/run_{current_time}')

# Set fixed action for the good agent
def fixed_action_policy(observations, agent_name):
    if "agent" in agent_name:
        return 0  # Action corresponding to "no movement"
    else:
        raise ValueError("Fixed action policy should only be used for good agents.")

# Model hyperparameters
n_agents = 1  # Only training the adversary
obs_dim = env.observation_space(env.agents[0]).shape[0]
action_dim = env.action_space(env.agents[0]).n
state_dim = 37
n_block = 3
n_embd = 32
n_head = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_args_dict = {
    "episode_length": 100,
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
    "lr": 0.0007,
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
    shared_obs, obs, rewards, dones1, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

    dones = []
    dones.append(dones1)
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

def train():
    """Train adversary policies with data in buffer. """
    trainer.prep_training()
    train_infos = trainer.train(buffer)      
    buffer.after_update()
    return train_infos

# Training function
def train_adversary(env, episodes=1000, gamma=0.99, save_interval=100, save_path=f'models/adversary_{current_time}.pth'):
    for episode in range(episodes):
        observations, infos = env.reset()

        obs_tensor = torch.tensor([observations[env.agents[0]]], dtype=torch.float32, device=device).unsqueeze(0)
        state_tensor = torch.zeros((1, 1, state_dim), dtype=torch.float32, device=device)

        total_reward = 0
        # log_probs = []
        # values = []
        # rewards = []

        step = 0

        while env.agents:
            adversary = env.agents[0]
            good_agent = env.agents[1]

            # Get action for the adversary
            values_return, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = collect(step)

            action = actions[0, 0].item()
            log_prob = action_log_probs[0, 0]
            value = values_return[0, 0]

            # Manually set the good agent's action
            good_action = fixed_action_policy(observations, good_agent)

            # Step the environment with both agents
            actions_dict = {adversary: action, good_agent: good_action}
            observations, rewards_dict, terminations, truncations, infos = env.step(actions_dict)

            shared_obs = concat_observation(list(observations.values()))

            total_reward += rewards_dict[adversary]
            # rewards.append(rewards_dict[adversary])
            # log_probs.append(log_prob)
            # values.append(value)
            data_ad = shared_obs, observations[adversary], np.array([rewards_dict[adversary]]), terminations[adversary], infos[adversary], values_return, actions, action_log_probs, rnn_states, rnn_states_critic

            insert(data_ad)

            # Prepare inputs for the next step
            if adversary in observations:
                obs_tensor = torch.tensor([observations[adversary]], dtype=torch.float32, device=device).unsqueeze(0)
                state_tensor = torch.zeros((1, 1, state_dim), dtype=torch.float32, device=device)
            
            step += 1

        compute()
        train_info = train()

        writer.add_scalar(f'Reward/{adversary}', total_reward, episode)

        # Save the model periodically
        if (episode + 1) % save_interval == 0:
            # save_model(model, save_path)
            pass
        buffer_rewards = np.mean(buffer.rewards) * all_args.episode_length

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Buffer ReWARD: {buffer_rewards}")

    # Close TensorBoard writer
    writer.close()


def evaluate_adversary(env, model, eval_episodes=10, model_path=None):
    """
    Evaluate the trained adversary in the environment.
    
    Args:
        env: The environment to evaluate the agent in.
        model: The model to be evaluated.
        eval_episodes: Number of episodes to run the evaluation.
        model_path: Path to the trained model file to load.
    """
    if model_path:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print("No model path provided. Using current model.")

    model.eval()

    total_rewards = []
    for episode in range(eval_episodes):
        observations, infos = env.reset()
        done = {agent: False for agent in env.agents}

        obs_tensor = torch.tensor([observations[env.agents[0]]], dtype=torch.float32, device=device).unsqueeze(0)
        state_tensor = torch.zeros((1, 1, state_dim), dtype=torch.float32, device=device)

        episode_reward = 0

        while not all(done.values()):
            adversary = env.agents[0]
            good_agent = env.agents[1]

            # Get action for the adversary
            actions, _, _ = model.get_actions(state_tensor, obs_tensor, deterministic=True)
            action = actions[0, 0].item()

            # Manually set the good agent's action
            good_action = fixed_action_policy(observations, good_agent)

            # Step the environment
            actions_dict = {adversary: action, good_agent: good_action}
            observations, rewards_dict, terminations, truncations, infos = env.step(actions_dict)

            # Update episode reward
            episode_reward += rewards_dict.get(adversary, 0)

            # Render the environment for observation
            env.render()

            # Prepare inputs for the next step
            if adversary in observations:
                obs_tensor = torch.tensor([observations[adversary]], dtype=torch.float32, device=device).unsqueeze(0)
                state_tensor = torch.zeros((1, 1, state_dim), dtype=torch.float32, device=device)

            done = {agent: terminations.get(agent, False) or truncations.get(agent, False) for agent in env.agents}

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{eval_episodes}, Reward: {episode_reward:.2f}")

    average_reward = np.mean(total_rewards)
    print(f"Evaluation completed over {eval_episodes} episodes. Average Reward: {average_reward:.2f}")

    # Close the environment
    env.close()

# Specify the path to the latest saved model
# latest_model_path = f'models/adversary_Jan03_00-50-05_ep1000.pth'

# Evaluate the trained adversary
# evaluate_adversary(env, model, eval_episodes=10, model_path=latest_model_path)

# Train the adversary
train_adversary(env, episodes=5000)