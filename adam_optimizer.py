import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import json
import os


class Policy(nn.Module):
    def __init__(self, state_dim=4, action_dim=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(state_dim, action_dim, dtype=torch.float16))
        self.bias = nn.Parameter(torch.randn(1, action_dim, dtype=torch.float16))
    
    def forward(self, state):
        return torch.matmul(state.float(), self.weight.float()) + self.bias.float()


def rollout(policy, env, device, max_steps=1000):
    """Run one episode, collect rewards from gym at each step."""
    state, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0
    
    with torch.no_grad():
        while not done and steps < max_steps:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action = policy(state_t).squeeze().cpu().numpy()
            action = np.array([np.clip(action, -1, 1)])
            
            # Send action to gym, get reward back
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
    
    return total_reward


def eval_policy(policy, device, num_ep=5):
    """Evaluate policy by running episodes and collecting rewards."""
    env = gym.make("InvertedPendulum-v5")
    rewards = [rollout(policy, env, device) for _ in range(num_ep)]
    env.close()
    return np.mean(rewards)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    policy = Policy().to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1, weight_decay=0)
    
    best_reward = -np.inf
    best_params = None
    
    for it in range(200):
        # Sample multiple random parameter perturbations and evaluate
        rewards = []
        perturbations = []
        
        for _ in range(20):  # Try 20 different parameter sets
            # Randomly perturb parameters
            with torch.no_grad():
                original_params = [p.clone() for p in policy.parameters()]
                
                for p in policy.parameters():
                    p.add_(torch.randn_like(p) * 0.5)  # Large exploration
                
                # Evaluate this perturbed policy
                reward = eval_policy(policy, device, num_ep=3)
                rewards.append(reward)
                perturbations.append([p.clone() for p in policy.parameters()])
                
                # Restore for next perturbation
                for i, p in enumerate(policy.parameters()):
                    p.copy_(original_params[i])
        
        # Find best perturbation
        best_idx = np.argmax(rewards)
        best_reward_iter = rewards[best_idx]
        
        if best_reward_iter > best_reward:
            best_reward = best_reward_iter
            best_params = {k: v.cpu().detach().numpy().tolist() for k, v in policy.named_parameters()}
        
        # Update policy towards best found
        with torch.no_grad():
            for p, best_p in zip(policy.parameters(), perturbations[best_idx]):
                p.copy_(best_p)
        
        # Clip params
        with torch.no_grad():
            for p in policy.parameters():
                p.clamp_(-6.0, 6.0)
        
        if it % 20 == 0:
            print(f"Iter {it:3d} | Reward: {best_reward_iter:7.2f} | Best: {best_reward:7.2f}")
    
    print(f"\nBest reward: {best_reward:.2f}")
    print(f"Best params: {best_params}")
    
    os.makedirs("logs/adam_optimizer", exist_ok=True)
    with open("logs/adam_optimizer/results.json", "w") as f:
        json.dump({
            "best_reward": float(best_reward),
            "best_params": best_params
        }, f, indent=2)


if __name__ == "__main__":
    train()

