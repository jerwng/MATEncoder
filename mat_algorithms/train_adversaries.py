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
import os

n_good_agents = 1
n_adversaries = 3
n_agents = n_adversaries  # Only training the adversary

# Initialize the environment with only 1 adversary and 1 good agent
def custom_env():
    env = simple_tag_v3.parallel_env(num_good=n_good_agents, num_adversaries=n_adversaries, num_obstacles=1, max_cycles=100, render_mode="human")
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

# Flag to store current direction of rule based action, -1 = left/down, 1 = right/up
current_rule_based_action_x = -1 
current_rule_based_action_y = -1

def rule_based_action_policy(observations, agent_name):
    global current_rule_based_action_x
    global current_rule_based_action_y

    direction = np.random.choice(['x', 'y'])

    # Indices of x and y positions in the observations array
    agent_x_position_state_index = 2
    agent_y_position_state_index = 3

    if "agent" in agent_name:
        if direction == 'x':
            if current_rule_based_action_x == -1:
                # Keep moving left until agent position state decreases to -0.75
                if observations[agent_name][agent_x_position_state_index] > -0.75:
                    return 1 # Action corresponding to move left
                else:
                    current_rule_based_action_x = 1
                    return 2
            else:
                # Keep moving right until agent position state increases to 0.75
                if observations[agent_name][agent_x_position_state_index] < 0.75:
                    return 2 # Action corresponding to move right
                else:
                    current_rule_based_action_x = -1
                    return 1
        
        if direction == 'y':
            if current_rule_based_action_y == -1:
                # Keep moving left until agent position state decreases to -0.75
                if observations[agent_name][agent_y_position_state_index] > -0.75:
                    return 3 # Action corresponding to move down
                else:
                    current_rule_based_action_y = 1
                    return 4
            else:
                # Keep moving right until agent position state increases to 0.75
                if observations[agent_name][agent_y_position_state_index] < 0.75:
                    return 4 # Action corresponding to move up
                else:
                    current_rule_based_action_y = -1
                    return 3
    else:
        raise ValueError("Fixed action policy should only be used for good agents.")

# Model hyperparameters
obs_dim = env.observation_space(env.agents[0]).shape[0]
action_dim = env.action_space(env.agents[0]).n
state_dim = 37
n_block = 4
n_embd = 512
n_head = 8
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

# Directory to save models
save_dir = f"models/"
os.makedirs(save_dir, exist_ok=True)

# Directory to load model
model_dir = f"models/transformer_Feb20_18-15-33.pt"

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

    dones = np.array(dones1).reshape(1, n_agents)

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
def train_adversary(env, episodes=1000, gamma=0.99, save_interval=100):
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
            adversaries = []
            good_agents = []

            for i in range(n_adversaries):
                adversaries.append(env.agents[i])
            
            for i in range(n_adversaries, n_adversaries + n_good_agents):
                good_agents.append(env.agents[i])

            # Get action for the adversary
            values_return, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = collect(step)

            adversary_actions = []
            for i in range(n_adversaries):
                adversary_actions.append(actions[0, i].item())

            good_agent_actions = []
            for i in range(n_good_agents):
                # Manually set the good agent's action
                # good_agent_actions.append(fixed_action_policy(observations, good_agents[i]))
                good_agent_actions.append(rule_based_action_policy(observations, good_agents[i]))

            # Step the environment with both agents
            actions_dict = {}

            for i in range(n_adversaries):
                actions_dict[adversaries[i]] = adversary_actions[i]
            
            for i in range(n_good_agents):
                actions_dict[good_agents[i]] = good_agent_actions[i]

            observations, rewards_dict, terminations, truncations, infos = env.step(actions_dict)

            shared_obs = concat_observation(list(observations.values()))

            for i in range(n_adversaries):
                total_reward += rewards_dict[adversaries[i]]

            # rewards.append(rewards_dict[adversary])
            # log_probs.append(log_prob)
            # values.append(value)

            obs_ad_list = []
            rewards_dict_list = []
            terminations_list = []
            infos_list = []

            for i in range(n_adversaries):
                obs_ad_list.append(observations[adversaries[i]])
                rewards_dict_list.append(rewards_dict[adversaries[i]])
                terminations_list.append(terminations[adversaries[i]])
                infos_list.append(infos[adversaries[i]])

            obs_ad = np.array(obs_ad_list)
            rewards_ad = np.array(rewards_dict_list).reshape(n_agents,1)
            term_ad = np.array(terminations_list).reshape(n_agents,1)
            info_ad = np.array(infos_list)

            # data_ad = shared_obs, observations[adversary1], np.array([rewards_dict[adversary1]]), terminations[adversary1], infos[adversary1], values_return, actions, action_log_probs, rnn_states, rnn_states_critic
            data_ad = shared_obs, obs_ad, rewards_ad, term_ad, info_ad, values_return, actions, action_log_probs, rnn_states, rnn_states_critic

            insert(data_ad)

            # Prepare inputs for the next step
            # if adversary1 in observations:
            #     obs_tensor = torch.tensor([observations[adversary]], dtype=torch.float32, device=device).unsqueeze(0)
            #     state_tensor = torch.zeros((1, 1, state_dim), dtype=torch.float32, device=device)
            
            step += 1

        compute()
        train_info = train()

        for i in range(n_adversaries):
            writer.add_scalar(f'Reward/{adversaries[i]}', total_reward, episode)

        # Save the model periodically
        if (episode % save_interval == 0 or episode == episodes - 1):
            save()

        buffer_rewards = np.mean(buffer.rewards) * all_args.episode_length

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Buffer ReWARD: {buffer_rewards}")

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
            adversaries = []
            good_agents = []

            for i in range(n_adversaries):
                adversaries.append(env.agents[i])
            
            for i in range(n_adversaries, n_adversaries + n_good_agents):
                good_agents.append(env.agents[i])

            shared_obs = concat_observation(list(observations.values()))
            shared_obs = np.tile(shared_obs, (n_adversaries, 1)) 

            obs_ad_list = []

            for i in range(n_adversaries):
                obs_ad_list.append(observations[adversaries[i]])
               
            obs = np.stack(obs_ad_list, axis=0)

            # Get action for the adversary
            trainer.prep_rollout()
            actions, _ = trainer.policy.act(shared_obs, obs, state_tensor, masks, deterministic=True)

            adversary_actions = []
            for i in range(n_adversaries):
                adversary_actions.append(actions[i,0].item())

            good_agent_actions = []
            for i in range(n_good_agents):
                # Manually set the good agent's action
                # good_agent_actions.append(fixed_action_policy(observations, good_agents[i]))
                good_agent_actions.append(rule_based_action_policy(observations, good_agents[i]))

            # Step the environment with both agents
            actions_dict = {}

            for i in range(n_adversaries):
                actions_dict[adversaries[i]] = adversary_actions[i]
            
            for i in range(n_good_agents):
                actions_dict[good_agents[i]] = good_agent_actions[i]

            observations, rewards_dict, terminations, truncations, infos = env.step(actions_dict)

            # Render the environment
            env.render()

            for i in range(n_adversaries):
                total_reward += rewards_dict[adversaries[i]]

            # Prepare inputs for the next step
            # if adversary in observations:
                # obs_tensor = torch.tensor([observations[adversary]], dtype=torch.float32, device=device).unsqueeze(0)

        total_rewards.append(total_reward)
        print(f"Evaluation Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {episodes} Evaluation Episodes: {avg_reward:.2f}")

# Evaluate the trained adversary
evaluate(env, episodes=10)

# Train the adversary
# train_adversary(env, episodes=5000)