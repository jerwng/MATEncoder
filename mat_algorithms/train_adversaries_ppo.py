from pettingzoo.mpe import simple_tag_v3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from datetime import datetime
import os

# Initialize the environment with only 1 adversary and 1 good agent
def custom_env():
    env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=1, num_obstacles=1, max_cycles=25, render_mode=None)
    env.reset()

    # Keep only one adversary and one good agent
    adversary = [agent for agent in env.agents if "adversary" in agent][0]
    good_agent = [agent for agent in env.agents if "agent" in agent][0]
    env.agents = [adversary, good_agent]

    return env

# Wrapping environment to make it compatible with Stable-Baselines3
class SingleAgentEnvWrapper(gym.Env):
    def __init__(self, env):
        super(SingleAgentEnvWrapper, self).__init__()
        self.env = env
        self.agent = env.agents[0]  # Train only the adversary
        self.action_space = env.action_space(self.agent)
        self.observation_space = env.observation_space(self.agent)

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset()
        return observations[self.agent], {}

    def step(self, action):
        # Set fixed action for the good agent
        good_agent = self.env.agents[1]
        good_action = 0  # Action corresponding to "no movement"

        # Step the environment
        actions = {self.agent: action, good_agent: good_action}
        observations, rewards, terminations, truncations, infos,  = self.env.step(actions)

        obs = observations[self.agent]
        reward = rewards[self.agent]
        done = terminations[self.agent]
        truncated = truncations[self.agent]
        info = infos[self.agent]

        return obs, reward, done, truncated, info

    # def render(self, mode="human"):
    #     self.env.render()

    def close(self):
        self.env.close()

# Custom callback to log rewards
class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_dir, verbose=1):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.step_rewards = []
        self.writer = None

    def _on_training_start(self):
        # Set up TensorBoard writer
        self.writer = configure(self.log_dir, ["tensorboard"])

    def _on_step(self):
        # Log reward for each step
        reward = self.locals["rewards"][0]
        self.step_rewards.append(reward)
        self.logger.record("reward/step", reward)
        self.episode_rewards.append(reward)
        return True

    def _on_rollout_end(self):
        # Log the total reward of the episode
        total_reward = sum(self.episode_rewards)
        self.logger.record("reward/total", total_reward)
        self.episode_rewards = []
        self.step_rewards = []

    def _on_training_end(self):
        if self.writer:
            self.writer.close()

# Create custom environment
env = custom_env()
wrapped_env = SingleAgentEnvWrapper(env)

# Use VecEnv for Stable-Baselines3 compatibility
vec_env = DummyVecEnv([lambda: wrapped_env])

# Define current time for logging and model saving
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
log_dir = f"logs/run_{current_time}"
os.makedirs(log_dir, exist_ok=True)

# Directory to save models
save_dir = f"models/"
os.makedirs(save_dir, exist_ok=True)

# Initialize PPO model
model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=1e-3,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)

# Train PPO model with logging callback
def train_with_ppo(model, total_timesteps=5000, save_interval=1000):
    callback = RewardLoggingCallback(log_dir=log_dir)
    timesteps = 0
    while timesteps < total_timesteps:
        # Train for the next interval
        model.learn(total_timesteps=save_interval, reset_num_timesteps=False, callback=callback)
        timesteps += save_interval

        # Save model periodically
        model.save(os.path.join(save_dir, f"ppo_adversary_{timesteps}"))

    print("Training completed!")
    model.save(os.path.join(save_dir, "ppo_adversary_final_{current_time}"))

# Train the model
train_with_ppo(model, total_timesteps=50000, save_interval=10000)
