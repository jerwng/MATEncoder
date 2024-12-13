import torch
import numpy as np
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer

# Define the input dimensions
state_dim = 37        # State dimension (seems fixed at 37)
obs_dim = 64          # Observation dimension
action_dim = 9        # Action dimension
n_agent = 3           # Number of agents 
n_block = 2           # Number of Transformer blocks
n_embd = 64           # Embedding dimension
n_head = 4            # Number of attention heads
encode_state = False  # Whether to encode the state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_type = 'Discrete'  # Action type
dec_actor = False     # Whether to use decoder actor
share_actor = False   # Whether to share actor parameters

# Initialize the model
model = MultiAgentTransformer(state_dim, obs_dim, action_dim, n_agent, n_block, n_embd, n_head,
                              encode_state=encode_state, device=device, action_type=action_type,
                              dec_actor=dec_actor, share_actor=share_actor)

# Generate random inputs
batch_size = 8  # Number of batches

state = np.random.rand(batch_size, n_agent, state_dim).astype(np.float32)
obs = np.random.rand(batch_size, n_agent, obs_dim).astype(np.float32)
action = np.random.randint(0, action_dim, (batch_size, n_agent, 1)).astype(np.int64)
available_actions = np.random.randint(0, 2, (batch_size, n_agent, action_dim)).astype(np.float32)

# print(f"Random state: {state}\n")
# print("----------------------------------------------------------------------------------\n")
# print(f"Random obs: {obs}\n")
# print("----------------------------------------------------------------------------------\n")
# print(f"Random action: {action}\n")
# print("----------------------------------------------------------------------------------\n")
# print(f"Random avail_actions: {available_actions}\n")

# Forward pass
action_log, v_loc, entropy = model.forward(state, obs, action, available_actions)
print(f"Action Logits: {action_log}\n")
print(f"Value Location: {v_loc}\n")
print(f"Entropy: {entropy}\n")

# Get actions
output_action, output_action_log, v_loc = model.get_actions(state, obs, available_actions, deterministic=False)
print(f"Output Actions: {output_action}\n")
print(f"Output Action Log: {output_action_log}\n")
print(f"Value Location: {v_loc}\n")

# Get values
values = model.get_values(state, obs, available_actions)
print(f"Values: {values}")
