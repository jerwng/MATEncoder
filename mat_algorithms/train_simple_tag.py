from pettingzoo.mpe import simple_tag_v3
import torch
import numpy as np
from torch.distributions import Categorical
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from datetime import datetime

# Initialize the environment
env = simple_tag_v3.env(render_mode=None)
env.reset()
current_time = datetime.now().strftime("%b%d_%H-%M-%S")

# Extract environment parameters
n_agents = len(env.agents)  # Number of agents
obs_dim = env.observation_space(env.agents[0]).shape[0]  # Observation space dimension
action_dim = env.action_space(env.agents[0]).n  # Number of discrete actions

# Model hyperparameters
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


def train(env, model, optimizer, episodes=1000, gamma=0.99, save_interval=100, save_path=f'models/transformer_{current_time}.pth'):
    torch.autograd.set_detect_anomaly(True)
    
    model.train()
    for episode in range(episodes):
        env.reset()
        done = False
        rewards = []
        log_probs = []
        values = []

        while not done:
            
            obs_all = []
            state_all = []
            done_all = []
            
            for i in range(n_agents):
                obs, reward, termination, truncation, _ = env.last()
                done = termination or truncation

                # if done:
                #     env.step(None)  # Pass a null action for terminated agents
                #     continue

                # Convert observation to tensor
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                state_tensor = torch.zeros(obs_tensor.shape[:-1] + (state_dim,), dtype=torch.float32).to(device)


                obs_all.append(obs_tensor)
                state_all.append(state_tensor)

                # # Get the action index and step the environment
                # action_index = action.squeeze().cpu().numpy()
                # env.step(action_index)

                # # Save the log probability and value
                # log_probs.append(action_log_prob)
                # values.append(value)
                if not done:
                    rewards.append(reward)

                done_all.append(done)
        
            obs_all_tensor = torch.cat(obs_all, dim=1)
            state_all_tensor = torch.cat(state_all, dim=1)

            actions, action_log_probs, values_return = model.get_actions(state_all_tensor, obs_all_tensor, deterministic=False)

            squeezed_actions = actions.squeeze()
            squeezed_action_log_probs = action_log_probs.squeeze(dim=0)
            squeezed_values_return = values_return.squeeze(dim=0)

            for i in range(n_agents):
                done = done_all[i]

                if done:
                    env.step(None)
                    continue

                action_index = squeezed_actions[i].cpu().numpy()
                action_log_prob = squeezed_action_log_probs[i]
                value = squeezed_values_return[i]

                env.step(action_index)

                log_probs.append(action_log_prob)
                values.append(value)

        
        # Compute returns and update model
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        log_probs = torch.cat(log_probs).to(device)
        values = torch.cat(values).squeeze()

        # Compute loss: Policy loss + Value loss
        advantage = returns - values
        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {sum(rewards):.2f}")

        # Save the model at regular intervals
        if (episode + 1) % save_interval == 0:
            save_model(model, save_path)

# Run the training loop
train(env, model, optimizer, episodes=1000)
