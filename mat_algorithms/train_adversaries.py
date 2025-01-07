from pettingzoo.mpe import simple_tag_v3
import torch
import numpy as np
from torch.distributions import Categorical
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Initialize the environment with only 1 adversary and 1 good agent
def custom_env():
    env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=1, num_obstacles=1, max_cycles=50, render_mode='human')
    env.reset()

    # Keep only one adversary and one good agent
    adversary = [agent for agent in env.agents if "adversary" in agent][0]
    good_agent = [agent for agent in env.agents if "agent" in agent][0]
    env.agents = [adversary, good_agent]

    return env

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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

# Training function
def train_adversary(env, model, optimizer, episodes=1000, gamma=0.99, save_interval=100, save_path=f'models/adversary_{current_time}.pth'):
    model.train()
    for episode in range(episodes):
        observations, infos = env.reset()

        obs_tensor = torch.tensor([observations[env.agents[0]]], dtype=torch.float32, device=device).unsqueeze(0)
        state_tensor = torch.zeros((1, 1, state_dim), dtype=torch.float32, device=device)

        total_reward = 0
        log_probs = []
        values = []
        rewards = []

        while env.agents:
            adversary = env.agents[0]
            good_agent = env.agents[1]

            # Get action for the adversary
            actions, action_log_probs, values_return = model.get_actions(state_tensor, obs_tensor, deterministic=False)
            action = actions[0, 0].item()
            log_prob = action_log_probs[0, 0]
            value = values_return[0, 0]

            # Manually set the good agent's action
            good_action = fixed_action_policy(observations, good_agent)

            # Step the environment with both agents
            actions_dict = {adversary: action, good_agent: good_action}
            observations, rewards_dict, terminations, truncations, infos = env.step(actions_dict)

            total_reward += rewards_dict[adversary]
            rewards.append(rewards_dict[adversary])
            log_probs.append(log_prob)
            values.append(value)

            # Prepare inputs for the next step
            if adversary in observations:
                obs_tensor = torch.tensor([observations[adversary]], dtype=torch.float32, device=device).unsqueeze(0)
                state_tensor = torch.zeros((1, 1, state_dim), dtype=torch.float32, device=device)

        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze()

        # Compute loss
        advantage = returns - values
        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss

        # Update the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log to TensorBoard
        writer.add_scalar(f'Reward/{adversary}', total_reward, episode)
        writer.add_scalar('Loss/Policy', policy_loss.item(), episode)
        writer.add_scalar('Loss/Value', value_loss.item(), episode)
        writer.add_scalar('Loss/Total', loss.item(), episode)

        # Save the model periodically
        if (episode + 1) % save_interval == 0:
            save_model(model, save_path)

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")

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
train_adversary(env, model, optimizer, episodes=5000)
