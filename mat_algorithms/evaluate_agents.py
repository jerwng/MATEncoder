import torch
from pettingzoo.mpe import simple_tag_v3
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
import numpy as np
from datetime import datetime
import time

# Model evaluation script

# Specify the trained model path
MODEL_PATH = "models/transformer_Dec20_01-08-46.pth"

# Initialize the environment
env = simple_tag_v3.parallel_env(render_mode="human")
observations, infos = env.reset()

# Extract environment parameters
n_agents = len(env.agents)  # Number of agents
obs_dim = env.observation_space(env.agents[0]).shape[0]  # Observation space dimension
action_dim = env.action_space(env.agents[0]).n  # Number of discrete actions

# Model hyperparameters (match these with training)
state_dim = 37 
n_block = 3
n_embd = 512
n_head = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = MultiAgentTransformer(
    state_dim=state_dim,
    obs_dim=obs_dim,
    action_dim=action_dim,
    n_agent=n_agents,
    n_block=n_block,
    n_embd=n_embd,
    n_head=n_head,
    encode_state=False,
    device=device,
    action_type='Discrete'
)

# Load the trained model
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def pad_to_length(arr, target_length, padding_value=0):
    """Pads an array to a target length with a padding value."""
    return np.pad(arr, (0, max(0, target_length - len(arr))), constant_values=padding_value)

def obs_dict_to_tensor(agents, obs_dict, target_dimension):
    """Converts observation dictionary to tensor for input to the model."""
    obs_list = []
    for agent in agents:
        padded_data = pad_to_length(obs_dict[agent], target_dimension)
        obs_list.append(torch.tensor(padded_data).unsqueeze(0).unsqueeze(0).to(device))
    return torch.cat(obs_list, dim=1).to(device)

def action_tensor_to_dict(agents, action_tensor):
    """Converts action tensor to a dictionary format required by the environment."""
    action_dict = {}
    for i, agent in enumerate(agents):
        action_dict[agent] = np.int64(action_tensor[i].item())
    return action_dict

def evaluate(env, model, episodes=10):
    """Evaluates the trained model and renders the environment."""
    for episode in range(episodes):
        observations, infos = env.reset()

        # Render the initial state
        env.render()
        # time.sleep(0.5)

        # Start evaluation loop
        total_rewards = {agent: 0 for agent in env.agents}

        while env.agents:
            # Convert observations to tensor format
            observation_tensor = obs_dict_to_tensor(env.agents, observations, obs_dim)
            state_tensor = torch.zeros(observation_tensor.shape[:-1] + (state_dim,), dtype=torch.float32).to(device)

            # Get actions from the model
            actions, _, _ = model.get_actions(state_tensor, observation_tensor, deterministic=True)

            # Convert actions to environment format
            actions_dict = action_tensor_to_dict(env.agents, actions.squeeze())

            # Step the environment
            observations, rewards, terminations, truncations, infos = env.step(actions_dict)

            # Render the environment
            env.render()
            # time.sleep(0.01)  # Slow down rendering for better visualization

            # Update total rewards
            for agent, reward in rewards.items():
                total_rewards[agent] += reward

        # Log the episode results
        print(f"Episode {episode + 1}/{episodes} - Total Rewards: {total_rewards}")

# Run the evaluation
evaluate(env, model, episodes=5)

# Close the environment
env.close()
