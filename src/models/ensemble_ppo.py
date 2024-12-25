# ensemble_ppo.py

import numpy as np
import torch
from models import ppo 

class EnsemblePPO:
    def __init__(self, num_agents, **ppo_kwargs):

        self.num_agents = num_agents
        self.agents = [ppo.PPO(**ppo_kwargs) for _ in range(num_agents)]

    def select_action(self, states):

        actions_list = []
        logprobs_list = []
        state_values_list = []

        for agent in self.agents:
            actions, logprobs, state_values = agent.select_action(states)
            actions_list.append(actions)
            logprobs_list.append(logprobs)
            state_values_list.append(state_values)

        actions_array = np.array(actions_list)  
        
        if not self.agents[0].has_continuous_action_space:
            aggregated_actions = []
            for i in range(actions_array.shape[1]):  
                actions_for_state = actions_array[:, i]
                counts = np.bincount(actions_for_state)
                aggregated_action = np.argmax(counts)
                aggregated_actions.append(aggregated_action)
            aggregated_actions = np.array(aggregated_actions)
        else:
            aggregated_actions = np.mean(actions_array, axis=0)

        aggregated_logprobs = torch.stack(logprobs_list).mean(dim=0)
        aggregated_state_values = torch.stack(state_values_list).mean(dim=0)

        return aggregated_actions, aggregated_logprobs, aggregated_state_values

    def select_action(self, states):
        actions_list = []
        logprobs_list = []
        state_values_list = []

        for agent in self.agents:
            actions, logprobs, state_values = agent.select_action(states)
            actions_list.append(actions)
            logprobs_list.append(logprobs)
            state_values_list.append(state_values)

        return actions_list, logprobs_list, state_values_list
    
    def evaluate(self, states, actions):

        logprobs_list = []
        state_values_list = []
        dist_entropy_list = []

        for agent in self.agents:
            logprobs, state_values, dist_entropy = agent.policy.evaluate(states, actions)
            logprobs_list.append(logprobs)
            state_values_list.append(state_values)
            dist_entropy_list.append(dist_entropy)

        aggregated_logprobs = torch.stack(logprobs_list).mean(dim=0)
        aggregated_state_values = torch.stack(state_values_list).mean(dim=0)
        aggregated_dist_entropy = torch.stack(dist_entropy_list).mean(dim=0)

        return aggregated_logprobs, aggregated_state_values, aggregated_dist_entropy

    def update(self):

        for agent in self.agents:
            agent.update()

    def save(self, path_prefix):

        for idx, agent in enumerate(self.agents):
            agent.save(f"{path_prefix}_agent_{idx}.pth")

    def load(self, path_prefix):

        for idx, agent in enumerate(self.agents):
            agent.load(f"{path_prefix}_agent_{idx}.pth")
            print(f"Agent {idx} loaded successfully.")
