import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  
from utils import fem
import logging

class BeamOptimizationEnv(gym.Env):
    metadata = {'render.modes': ['human']}  
    def __init__(self, width=4, height=4, density=0.4, step_size=0.5, optimal_density=0.5, reward_weights = None, beam_type=1 ):
        super(BeamOptimizationEnv, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.width = width
        self.height = height
        self.density = density
        self.step_size = step_size  
        self.optimal_density = optimal_density  
        self.max_steps = self.width * self.height * 2 #* 5
        self.action_space = gym.spaces.Discrete(2 * self.width * self.height)
        self.state_dim = self.width*self.height  + (self.width + 1)*(self.height + 1) * 2
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.state_dim, ), dtype=np.float32)
        self.step_size_initial = 1.0
        self.step_size_final = 0.05
        self.best_compliance = torch.full((10,), float('inf'), dtype=torch.float32)
        self.beam_type = beam_type
        self.phase_threshold_1 = 0.05
        self.phase_threshold_2 = 0.1
        # self.reset()

        beam_functions = {
            1: fem.mbb_beam_1,
            2: fem.mbb_beam_2,
            3: fem.mbb_beam_3,
            4: fem.mbb_beam_4,
            5: fem.mbb_beam_5,
            6: fem.mbb_beam_6,
            7: fem.mbb_beam_7,
            8: fem.mbb_beam_8,
            9: fem.mbb_beam_9

        }
        
        # normals, forces, _ = fem.mbb_beam(width, height, density)
        normals, forces, _ = beam_functions[beam_type](width, height, density)
        self.normals = torch.tensor(normals, dtype=torch.float32, device=self.device)
        self.forces = torch.tensor(forces, dtype=torch.float32, device=self.device)
        self.args = fem.get_args(self.normals.cpu().numpy(), self.forces.cpu().numpy(), density)
        
        self.current_compliance = float('inf')
        self.previous_compliance = float('inf')
        self.current_constraint = 0.5 
        self.reward = 0
        self.reset()

        if reward_weights == None:
            self.w_compliance = 1.0
            self.w_density_high = 1.0   
            self.w_density_low = 1.0    
            self.w_total_mass = 1.0
            self.w_entropy = 0.5
        else:
            self.w_compliance = reward_weights['w_compliance']
            self.w_density_high = reward_weights['w_density_high']  
            self.w_density_low = reward_weights['w_density_low']    
            self.w_total_mass = reward_weights['w_total_mass']
            self.w_entropy = reward_weights['w_entropy']

        self.weight_matrices = {
            7: self.compute_weight_matrix(size=7),
            5: self.compute_weight_matrix(size=5),
            3: self.compute_weight_matrix(size=3)
        }

        self.base_compliance = {
            1: fem.optim(fem.get_args(*fem.mbb_beam_1()), x= None)[0],
            2: fem.optim(fem.get_args(*fem.mbb_beam_2()), x= None)[0],
            3: fem.optim(fem.get_args(*fem.mbb_beam_3()), x= None)[0],
            4: fem.optim(fem.get_args(*fem.mbb_beam_4()), x= None)[0],
            5: fem.optim(fem.get_args(*fem.mbb_beam_5()), x= None)[0],
            6: fem.optim(fem.get_args(*fem.mbb_beam_6()), x= None)[0],
            7: fem.optim(fem.get_args(*fem.mbb_beam_7()), x= None)[0],
            8: fem.optim(fem.get_args(*fem.mbb_beam_8()), x= None)[0],
            9: fem.optim(fem.get_args(*fem.mbb_beam_9()), x= None)[0]

        }
    def construct_state(self, state, forces):

        if not isinstance(forces, torch.Tensor):
            raise TypeError("Forces must be a torch.Tensor")
        
        forces_flat = forces.flatten()
        
        concatenated_state = torch.cat((state, forces_flat), dim=0)
        if torch.isnan(concatenated_state).any():
            print("NaN detected in concatenated state!")
            concatenated_state = torch.nan_to_num(concatenated_state, nan=0.0)
        # print(f"shape od concatenadet state is {concatenated_state.shape}")
        return concatenated_state

    def reset(self,*, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        noise_std = 0.05

        # Create a temporary state: for each cell, sample from a Gaussian centered at 0.5.
        #temp_state = 0.5 + torch.randn((self.width * self.height), dtype=torch.float32, device=self.device) * noise_std
        temp_state = torch.ones((self.width*self.height), dtype=bool, device = self.device) * 0.5
        self.current_step = 0
        self.visited = torch.zeros((self.width * self.height), dtype=bool, device=self.device)
        self.visited_cells = set()
        self.current_compliance = float('inf')
        self.previous_compliance = float('inf')
        self.current_constraint = 1.0
        self.reward = 0
        #self.best_copmliance = float('inf')
        self.current_compliance = float('inf')
        self.state = self.construct_state(temp_state, self.forces)

        # print(f"shape of constructed space is {self.state.shape}")

        return self.state.cpu().numpy(),{}

    def check_connectivity(self, densities):
  
        grid = densities.reshape(self.height, self.width)  # shape (H,W)
        # Threshold to binary
        binary_mask = (grid > 0.2).astype(np.uint8)

        # run BFS/Union-Find
        visited = np.zeros_like(binary_mask, dtype=bool)
        def neighbors(r, c):
            for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                if 0<=nr<self.height and 0<=nc<self.width:
                    yield nr,nc

        components = []
        for r in range(self.height):
            for c in range(self.width):
                if binary_mask[r,c] == 1 and not visited[r,c]:
                    # BFS to find connected region
                    queue = [(r,c)]
                    visited[r,c] = True
                    size = 0
                    while queue:
                        rr,cc = queue.pop()
                        size += 1
                        for (rr2, cc2) in neighbors(rr, cc):
                            if binary_mask[rr2,cc2] == 1 and not visited[rr2,cc2]:
                                visited[rr2,cc2] = True
                                queue.append((rr2,cc2))
                    components.append(size)

        # Possibly define a penalty if there's more than 1 big component
        # or if there's a small floating component
        # For example:
        if len(components) <= 1:
            return 0.0  # no penalty
        else:
            # penalty based on how many or how big the extra comps are
            # e.g. sum of the sizes of all but the largest comp
            largest = max(components)
            others_sum = sum(components) - largest
            penalty = 0.1 * others_sum  # scale as you want
            return penalty

    def calculate_total_density(self):
        """
        Calculate the mean density of the current state.
        """
        return np.mean(self.state.cpu().numpy())

    def get_neighboring_cells(self, cell):
 
        neighbors = []
        row = cell // self.width
        col = cell % self.width
        movements = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1,-1), (1,1)]
        for dr, dc in movements:
            new_row = row + dr
            new_col = col + dc
            if 0 <= new_row < self.height and 0 <= new_col < self.width:
                neighbor = new_row * self.width + new_col
                neighbors.append(neighbor)

        return neighbors

    
    def compute_weight_matrix(self, size):
        """
        Compute a weight matrix for a given subgrid size where central cells have higher weights.
        
        Args:
            size (int): Size of the subgrid (e.g., 5 for 5x5).
        
        Returns:
            torch.Tensor: Weight matrix of shape (size, size).
        """
        center = size // 2
        weights = np.zeros((size, size), dtype=np.float32)
        max_distance = center
        for r in range(size):
            for c in range(size):
                distance = max(abs(r - center), abs(c - center))
                # Define weight based on distance
                if distance == 0:
                    weight = 1.0
                elif distance == 1:
                    weight = 0.8
                elif distance == 2:
                    weight = 0.6
                elif distance == 3:
                    weight = 0.4
                elif distance == 4:
                    weight = 0.2
                else:
                    weight = 0.0  # Beyond the subgrid
                weights[r, c] = weight
        # Normalize weights to ensure the maximum weight is 1
        weights /= weights.max()
        return torch.tensor(weights, dtype=torch.float32, device=self.device)
    def get_subgrid_cells(self, cell, size):

        subgrid_cells = []
        half_size = size // 2
        row = cell // self.width
        col = cell % self.width

        for dr in range(-half_size, half_size + 1):
            for dc in range(-half_size, half_size + 1):
                new_row = row + dr
                new_col = col + dc
                if 0 <= new_row < self.height and 0 <= new_col < self.width:
                    neighbor = new_row * self.width + new_col
                    subgrid_cells.append(neighbor)

        return subgrid_cells

    def step(self, action):
        """
        Perform the action and return the new state, reward, done, truncated, and info.
        """

        # --- 1) Map action to cell index + direction ---
        total_cells = self.width * self.height
        if action < 0 or action >= 2 * total_cells:
            raise ValueError(f"Invalid action: {action}")

        cell = action % total_cells  # which cell
        direction = 'increase' if action >= total_cells else 'decrease'

        # --- 2) Determine subgrid (phase) based on training progress ---
        progress = self.current_step / self.max_steps
        progress = np.clip(progress, 0.0, 1.0)


        phase1_steps = self.width * self.height  

        if self.current_step < phase1_steps:
            current_step_size = 1.0
            subgrid_size = 1
        else:
            # Second phase: linear decay from 1.0 to 0.05
            decay_progress = (self.current_step - phase1_steps) / (self.max_steps - phase1_steps)
            decay_progress = np.clip(decay_progress, 0.0, 1.0)
            current_step_size = 1.0 - (1.0 - 0.05) * decay_progress
            subgrid_size = 1

        # Ensure we never go below minimum step size
        current_step_size = max(current_step_size, 0.05)
        self.step_size = current_step_size
        # --- 3) Step size decay ---

        current_step_size = max(current_step_size, self.step_size_final)
        self.step_size = current_step_size

        compliance_before = self.previous_compliance

        reward_action = 0.0

            # Single cell
        old_density = self.state[cell].item()
        if direction == 'increase':
            new_density = min(old_density + self.step_size, 1.0)
        else:
            new_density = max(old_density - self.step_size, 0.001)

        if new_density != old_density:
            self.state[cell] = new_density
            reward_action += 0.01
        else:
            reward_action -= 0.01

        self.current_step += 1

        main_reward = 0.0
        # --- 6) Compute domain-specific reward (compliance, mass, etc.) ---
        #if (self.current_step % 5 ) == 0:
        main_reward, reward_info = self.calculate_reward() 
        # This sets self.current_compliance internally, so now we have the AFTER compliance.

        # --- 7) Intermediate compliance-improvement reward ---
        compliance_after = self.current_compliance
        self.previous_compliance = self.current_compliance

        reward = reward_action + main_reward 

        # Optionally, do mild scaling or clipping
        #reward = reward / (1 + abs(reward)) #
        reward = np.clip(reward, -1.0, 50.0)


        if self.current_compliance < self.best_compliance[self.beam_type] :
            self.best_compliance[self.beam_type] = self.current_compliance
            reward += 0.5  # small bonus for setting a new best
            #done = True

        done = (self.current_step >= self.max_steps)


        # Info dict
        info = {
            'compliance_before': compliance_before,
            'compliance_after': compliance_after,
            'reward_action': reward_action,
            'reward_main': main_reward,
            'compliance': self.current_compliance,
            'constraint': self.current_constraint,
            'subgrid_size': subgrid_size,
            'direction': direction,
            'step': self.current_step,
        }

        return self.state.cpu().numpy(), reward, done, False, info
    
    def calculate_reward(self):
        # Evaluate compliance, constraint, etc.
        with torch.no_grad():
            x = self.state.cpu().numpy()[: self.width * self.height]
            compliance, constraint = fem.optim(args=self.args, x=x)
        compliance_before = self.current_compliance
        self.current_compliance = compliance
        self.current_constraint = constraint


        compliance_improvement = 0.0
        if compliance < self.base_compliance[self.beam_type] and compliance_before < self.base_compliance[self.beam_type]:
                        # Positive reward when compliance is below baseline
                        compliance_improvement = 10 * (compliance_before - compliance) / compliance_before
                        compliance_improvement = (compliance_before - compliance) / compliance_before

        r_compliance = -np.log( compliance/ (self.base_compliance[self.beam_type]))
        r_compliance += compliance_improvement

        # print(f"compliance is {compliance} and constraint is {constraint}")
        densities = self.state[: self.width * self.height].cpu().numpy()
        total_mass = np.sum(densities)
        target_total_density = self.optimal_density * self.width * self.height
        mass_deviation = max(total_mass - target_total_density, 0)
        r_mass = -self.w_total_mass * mass_deviation

 
        # Sum them
        reward = (r_compliance + 0.2 * r_mass)
 

        info = {
            'raw_reward': reward,
            'compliance': compliance,
            'constraint': constraint,
            'r_compliance': r_compliance,
            'r_mass': r_mass,
        }

        return reward, info

    def step(self, action):
        """
        Perform the action and return the new state, reward, done, truncated, info.
        This version does NOT terminate the episode if compliance is large.
        Instead, we apply a soft penalty that we can later ramp up if needed.
        """

        # 1) Map action
        total_cells = self.width * self.height
        if action < 0 or action >= 2 * total_cells:
            raise ValueError(f"Invalid action: {action}")

        cell = action % total_cells
        direction = 'increase' if action >= total_cells else 'decrease'

        # 2) Step size logic
        phase1_steps = self.width * self.height
        if self.current_step < phase1_steps:
            current_step_size = 0.5  # example smaller step
        else:
            decay_progress = (self.current_step - phase1_steps) / (self.max_steps - phase1_steps)
            decay_progress = np.clip(decay_progress, 0.0, 1.0)
            current_step_size = 0.5 - (0.5 - 0.05) * decay_progress
        current_step_size = max(current_step_size, self.step_size_final)
        self.step_size = current_step_size

        # 3) Apply action
        old_density = self.state[cell].item()
        if direction == 'increase':
            new_density = min(old_density + self.step_size, 1.0)
        else:
            new_density = max(old_density - self.step_size, 0.01)

        # small immediate reward
        reward_action = 0.01 if new_density != old_density else -0.01
        self.state[cell] = new_density
        self.current_step += 1

        # 4) Compute domain-specific reward occasionally
        main_reward = 0.0
        if (self.current_step % 5) == 0:
            main_reward, reward_info = self.calculate_reward()
        else:
            reward_info = {}

        # combine rewards
        reward = reward_action + main_reward
        reward = np.clip(reward, -10.0, 10.0)  # or any range you prefer

        # 5) Optional bonus for new best compliance
        if self.current_compliance < self.best_compliance[self.beam_type]:
            self.best_compliance[self.beam_type] = self.current_compliance
            reward += 0.5

        # 6) Instead of terminating, apply a soft penalty for huge compliance
        if self.current_compliance > 1e5:
            reward -= 5.0  # big penalty, but still let the agent continue

        done = (self.current_step >= self.max_steps)
        truncated = False  # or any logic you want for truncation

        # info
        info = {
            'step': self.current_step,
            'reward_action': reward_action,
            'reward_main': main_reward,
            'compliance': self.current_compliance,
            'constraint': self.current_constraint,
            # etc...
        }
        info.update(reward_info)

        return self.state.cpu().numpy(), reward, done, truncated, info


    def calculate_reward(self):
        """
        Compute compliance, mass, etc. with a softer shaping.
        """
        with torch.no_grad():
            x = self.state.cpu().numpy()[:self.width * self.height]
            compliance, constraint = fem.optim(args=self.args, x=x)

        compliance_before = self.current_compliance
        self.current_compliance = compliance
        self.current_constraint = constraint

        baseline = self.base_compliance[self.beam_type]

        # 1) compliance ratio
        ratio = compliance / max(baseline, 1e-9)

        # 2) saturate ratio at some upper bound so it doesn't blow up your logs
        ratio_saturated = np.minimum(ratio, 1000.0)  # tune the saturate limit
        r_compliance = -np.log(ratio_saturated + 1e-9)

        # 3) small reward for improvement from previous step
        improvement = compliance_before - compliance
        if improvement > 0 and compliance_before < 1e9:
            # scale improvement by how big compliance was
            r_improvement = 0.05 * (improvement / (compliance_before + 1e-9))
        else:
            r_improvement = 0.0

        # 4) mass penalty
        densities = self.state[: self.width * self.height].cpu().numpy()
        total_mass = np.sum(densities)
        target_total = self.optimal_density * self.width * self.height
        mass_deviation = max(total_mass - target_total, 0)
        r_mass = -self.w_total_mass * mass_deviation * 0.2

        raw_reward = r_compliance + r_improvement + r_mass

        info = {
            'raw_reward': raw_reward,
            'compliance': compliance,
            'constraint': constraint,
            'r_compliance': r_compliance,
            'r_improvement': r_improvement,
            'r_mass': r_mass,
        }

        return raw_reward, info
    
    #'''


    def step(self, action):
        """
        Take an action in the environment. The action is an integer in the range
        [0, 2 * total_cells). The lower half corresponds to a 'decrease' action,
        and the upper half to an 'increase' action on the chosen cell.
        """
        total_cells = self.width * self.height
        if action < 0 or action >= 2 * total_cells:
            raise ValueError(f"Invalid action: {action}")

        # Determine which cell to update and in which direction
        cell = action % total_cells
        direction = 'increase' if action >= total_cells else 'decrease'

        # Compute progress in [0,1] for scheduling purposes
        progress = np.clip(self.current_step / self.max_steps, 0.0, 1.0)

        # --- Adaptive Step Size ---
        # Scale step size by the inverse square root of the grid area
        scaling_exponent = 0.25
        scaling_factor = total_cells ** scaling_exponent
        current_step_size = (self.step_size_initial -
                            (self.step_size_initial - self.step_size_final) * progress) / scaling_factor
        current_step_size = max(current_step_size, self.step_size_final / scaling_factor)
        self.step_size = current_step_size

        # Record the current compliance (if available) for later logging
        compliance_before = getattr(self, 'current_compliance', None)

        # --- Apply the Action ---
        # Update the density value of the selected cell
        reward_action = 0.0
        old_density = self.state[cell].item()
        if direction == 'increase':
            new_density = min(old_density + current_step_size, 1.0)
        else:
            new_density = max(old_density - current_step_size, 0.001)

        # Only reward if the cell value actually changes
        if new_density != old_density:
            self.state[cell] = new_density
            reward_action += 0.05  # small positive reward for making a change
        else:
            reward_action -= 0.05  # slight penalty if no change occurred

        self.current_step += 1

        # --- Calculate the Main Reward ---
        main_reward, reward_info = self.calculate_reward()

        # Combine the dense reward from the physical properties with the action reward.
        total_reward = reward_action + main_reward

        # --- Bonus for New Best Compliance ---
        # (Assuming lower compliance is better.)
        if self.current_compliance < self.best_compliance[self.beam_type]:
            self.best_compliance[self.beam_type] = self.current_compliance
            total_reward += 1.0  # bonus for setting a new best compliance

        done = (self.current_step >= self.max_steps)

        info = {
            'compliance_before': compliance_before,
            'compliance_after': self.current_compliance,
            'reward_action': reward_action,
            'reward_main': main_reward,
            'total_reward': total_reward,
            'current_step': self.current_step,
            'step_size': self.step_size,
            'reward_details': reward_info,
            'direction': direction,
            'cell_index': cell,
            'progress': progress,
        }

        return self.state.cpu().numpy(), total_reward, done, False, info


    def calculate_reward(self):
        """
        Compute a dense reward that considers the compliance (normalized by grid area),
        the deviation of the total mass from a target mass, and a connectivity penalty.
        """
        # Get the current state vector for the grid cells
        with torch.no_grad():
            x = self.state.cpu().numpy()[:self.width * self.height]
            compliance, constraint = fem.optim(args=self.args, x=x)
        if hasattr(self, 'current_compliance'):
            self.previous_compliance = self.current_compliance
        self.current_compliance = compliance
        self.current_constraint = constraint

        grid_area = self.width * self.height

        # If compliance is excessively high, return a heavy penalty immediately.


        # --- Compliance Reward ---
        # Normalize compliance by the grid area so that the term is size invariant.
        normalized_compliance = compliance / grid_area
        r_compliance = -self.w_compliance * normalized_compliance
        if compliance < self.base_compliance[self.beam_type]:
                # Calculate relative improvement (as a fraction of the baseline).
                if self.previous_compliance< self.base_compliance[self.beam_type]:
                    improvement = (self.previous_compliance - compliance) / self.previous_compliance 
                    r_compliance = -self.w_compliance * normalized_compliance + improvement
                else:
                    r_compliance = -self.w_compliance * normalized_compliance 
        else:

            # Penalize extra if the compliance is worse than baseline.
            r_compliance = -np.log(self.w_compliance * normalized_compliance)

                # --- Mass Penalty ---
        # Total mass is the sum of the densities; we want it to be near optimal_density * grid_area.
        densities = self.state[:grid_area].cpu().numpy()
        total_mass = np.sum(densities)
        target_mass = self.optimal_density * grid_area
        mass_error = total_mass - target_mass  # positive if too heavy, negative if too light
        # Normalize the mass error to an average per cell
        normalized_mass_error = mass_error / grid_area
        # Use a quadratic penalty to strongly discourage large deviations.
        r_mass = -self.w_total_mass * (normalized_mass_error ** 2)


        # --- Total Reward ---
        # Sum all the components.
        main_reward = r_compliance + r_mass 

        reward_info = {
            'compliance': compliance,
            'normalized_compliance': normalized_compliance,
            'r_compliance': r_compliance,
            'total_mass': total_mass,
            'target_mass': target_mass,
            'normalized_mass_error': normalized_mass_error,
            'r_mass': r_mass,
            'raw_reward': main_reward,
            'constraint': constraint,
        }

        return main_reward, reward_info

    def step(self, action):
        """
        Perform the action and return the new state, reward, done, truncated, and info.
        """
        print(f"we are in step okay")
        # --- 1) Map action to cell index + direction ---
        total_cells = self.width * self.height
        if action < 0 or action >= 2 * total_cells:
            raise ValueError(f"Invalid action: {action}")

        cell = action % total_cells  # which cell
        direction = 'increase' if action >= total_cells else 'decrease'

        # --- 2) Determine subgrid (phase) based on training progress ---
        progress = self.current_step / self.max_steps
        progress = np.clip(progress, 0.0, 1.0)

        if progress < self.phase_threshold_1:
            subgrid_size = 1
        elif progress < self.phase_threshold_2:
            subgrid_size = 1
        else:
            subgrid_size = 1

        # --- 3) Step size decay ---
        current_step_size = (
            self.step_size_initial
            - (self.step_size_initial - self.step_size_final) * progress
        )
        current_step_size = max(current_step_size, self.step_size_final)
        self.step_size = current_step_size

        # --- 4) Keep track of compliance BEFORE the action ---
        # If this is the very first step, self.current_compliance might not be set yet.
        # So handle that case gracefully:
        compliance_before = self.previous_compliance

        # --- 5) Apply the action (coarse or single-cell) ---
        reward_action = 0.0

            # Single cell
        old_density = self.state[cell].item()
        if direction == 'increase':
            new_density = min(old_density + self.step_size, 1.0)
        else:
            new_density = max(old_density - self.step_size, 0.001)

        if new_density != old_density:
            self.state[cell] = new_density
            reward_action += 0.05
        else:
            reward_action -= 0.05

        self.current_step += 1

        # --- 6) Compute domain-specific reward (compliance, mass, etc.) ---
        main_reward, reward_info = self.calculate_reward() 
        # This sets self.current_compliance internally, so now we have the AFTER compliance.

        compliance_after = self.current_compliance
        self.previous_compliance = self.current_compliance
        diff = 0
        #print(f"compliance currently is {compliance_after}")
        if compliance_before <  self.base_compliance[self.beam_type]:
            diff = compliance_before - compliance_after
            improvement_scale = 0.05
            reward_improvement = improvement_scale * diff / self.base_compliance[self.beam_type]
        else:
            reward_improvement = 0.0

        # Combine everything
        reward = reward_action + main_reward + reward_improvement

        # Optionally, do mild scaling or clipping
        reward = reward / (1 + abs(reward)) #np.clip(reward, -5.0, 5.0)


        if self.current_compliance < self.best_compliance[self.beam_type] :
            self.best_compliance[self.beam_type] = self.current_compliance
            reward += 0.1  # small bonus for setting a new best
            #done = True

        done = (self.current_step >= self.max_steps)
            #if done:
                #print(f"here motherfucke is done since {self.current_step}")

        # Info dict
        info = {
            'compliance_before': compliance_before,
            'compliance_after': compliance_after,
            'compliance_diff': diff ,
            'reward_improvement': reward_improvement,
            'reward_action': reward_action,
            'reward_main': main_reward,
            'compliance': self.current_compliance,
            'constraint': self.current_constraint,
            'subgrid_size': subgrid_size,
            'direction': direction,
            'step': self.current_step,
        }

        return self.state.cpu().numpy(), reward, done, False, info
  

    def calculate_reward(self):
        # Evaluate compliance, constraint, etc.
        with torch.no_grad():
            x = self.state.cpu().numpy()[: self.width * self.height]
            compliance, constraint = fem.optim(args=self.args, x=x)

        self.current_compliance = compliance
        self.current_constraint = constraint

        # If compliance is huge, let's clamp or heavily penalize it
        if compliance > 1e5:
            # or some large negative reward
            return -10.0, {'raw_reward': -100.0, 'compliance': compliance, 'constraint': constraint}

        penalty_connectivity = self.check_connectivity(x)

        # Weighted components
        r_compliance = -self.w_compliance * (compliance / self.base_compliance[1])
        r_compliance = -self.w_compliance * (compliance / self.width * self.height * self.base_compliance[self.beam_type])
        #r_compliance = 100.0 / ( compliance - 20.3)
        #print(f"r compliance is {r_compliance}")

        densities = self.state[: self.width * self.height].cpu().numpy()
  
        grey_mask = (densities > 0.2) & (densities < 0.8)
        fraction_grey = np.mean(grey_mask)  
        r_grey = -1* fraction_grey


        # Mass penalty or deviation
        total_mass = np.sum(densities)
        target_total_density = self.optimal_density * self.width * self.height
        mass_deviation = max(total_mass - target_total_density, 0)
        r_mass = -self.w_total_mass * mass_deviation


        reward = (
            r_compliance
            #+r_density_high
            #+ r_grey
            #+ r_density_low
            + 0.5 * r_mass
            #+ r_entropy
            #+ r_connectivity
        )
        # sprint(f"r compliance is {r_compliance} density  is {r_density_high} mass is {r_mass}")
        # Mild clip or no clip
        #reward = np.clip(reward, -100, 100)

        info = {
            'raw_reward': reward,
            'compliance': compliance,
            'constraint': constraint,
            'r_compliance': r_compliance,
            'r_mass': r_mass,
        }

        return reward, info

    def step(self, action):
        """
        Perform the action and return the new state, reward, done, truncated, and info.
        """

        # --- 1) Map action to cell index + direction ---
        total_cells = self.width * self.height
        if action < 0 or action >= 2 * total_cells:
            raise ValueError(f"Invalid action: {action}")

        cell = action % total_cells  # which cell
        direction = 'increase' if action >= total_cells else 'decrease'

        # --- 2) Determine subgrid (phase) based on training progress ---
        progress = self.current_step / self.max_steps
        progress = np.clip(progress, 0.0, 1.0)

        if progress < self.phase_threshold_1:
            subgrid_size = 1
        elif progress < self.phase_threshold_2:
            subgrid_size = 1
        else:
            subgrid_size = 1

        # --- 3) Step size decay ---
        current_step_size = (
            self.step_size_initial
            - (self.step_size_initial - self.step_size_final) * progress
        )
        current_step_size = max(current_step_size, self.step_size_final)
        self.step_size = current_step_size


        compliance_before = self.previous_compliance
        reward_action = 0.0

        # Single cell
        old_density = self.state[cell].item()
        if direction == 'increase':
            new_density = min(old_density + self.step_size, 1.0)
        else:
            new_density = max(old_density - self.step_size, 0.001)

        if new_density != old_density:
            self.state[cell] = new_density
            reward_action += 0.05
        else:
            reward_action -= 0.05

        self.current_step += 1

        main_reward = 0.0
        # --- 6) Compute domain-specific reward (compliance, mass, etc.) ---
        #if self.current_step % 2 == 0:
        main_reward, reward_info = self.calculate_reward() 
        # This sets self.current_compliance internally, so now we have the AFTER compliance.

        # --- 7) Intermediate compliance-improvement reward ---
        compliance_after = self.current_compliance
        self.previous_compliance = self.current_compliance
        diff = 0
        #print(f"compliance currently is {compliance_after}")
        if compliance_before <  self.base_compliance[self.beam_type]:
            diff = compliance_before - compliance_after
            # If diff > 0 => compliance improved
            # If diff < 0 => got worse
            # Scale it so we don't overshadow main_reward
            improvement_scale = 0.1 # or tweak as needed
            reward_improvement = improvement_scale * diff / self.base_compliance[self.beam_type]
        else:
            reward_improvement = 0.0

        # Combine everything
        reward = reward_action + main_reward + reward_improvement

        # Optionally, do mild scaling or clipping
        #reward = reward / (1 + abs(reward)) #np.clip(reward, -5.0, 5.0)


        if self.current_compliance < self.best_compliance[self.beam_type] :
            self.best_compliance[self.beam_type] = self.current_compliance
            reward += 0.5  # small bonus for setting a new best
            #done = True
        reward = reward / 10
        #reward = reward / (1 + abs(reward))
        reward = np.clip( reward, -2.0, 2.0)
        #reward = reward / 10.0

        done = (self.current_step >= self.max_steps)
            #if done:
                #print(f"here motherfucke is done since {self.current_step}")

        # Info dict
        info = {
            'compliance_before': compliance_before,
            'compliance_after': compliance_after,
            'compliance_diff': diff ,
            'reward_improvement': reward_improvement,
            'reward_action': reward_action,
            'reward_main': main_reward,
            'compliance': self.current_compliance,
            'constraint': self.current_constraint,
            'subgrid_size': subgrid_size,
            'direction': direction,
            'step': self.current_step,
        }

        return self.state.cpu().numpy(), reward, done, False, info
  

    def calculate_reward(self):
        # Evaluate compliance, constraint, etc.
        with torch.no_grad():
            x = self.state.cpu().numpy()[: self.width * self.height]
            compliance, constraint = fem.optim(args=self.args, x=x)

        self.current_compliance = compliance
        self.current_constraint = constraint
        total_cells = self.width * self.height
        # If compliance is huge, let's clamp or heavily penalize it
        #if compliance > 1e5:
            # or some large negative reward
        #    return -10.0, {'raw_reward': -100.0, 'compliance': compliance, 'constraint': constraint}

        r_compliance = -self.w_compliance * (compliance / self.base_compliance[1])

        densities = self.state[: self.width * self.height].cpu().numpy()

        total_mass = np.sum(densities)
        target_total_density = (self.optimal_density +0.05) * total_cells
        mass_deviation = max(total_mass - target_total_density, 0)
        r_mass = -self.w_total_mass * mass_deviation
        grey_mask = (densities > 0.2) & (densities < 0.8)
        fraction_grey = np.mean(grey_mask)  
        r_grey = -1* fraction_grey
   
        reward = (r_compliance + 0.5 * r_mass + r_grey )
   
        info = {
            'raw_reward': reward,
            'compliance': compliance,
            'constraint': constraint,
            'r_compliance': r_compliance,

            'r_mass': r_mass,

        }

        return reward, info

    def step(self, action):
            """
            Perform the action and return the new state, reward, done, truncated, and info.
            """

            # --- 1) Map action to cell index + direction ---
            total_cells = self.width * self.height
            if action < 0 or action >= 2 * total_cells:
                raise ValueError(f"Invalid action: {action}")

            cell = action % total_cells  # which cell
            direction = 'increase' if action >= total_cells else 'decrease'

            progress = self.current_step / self.max_steps
            progress = np.clip(progress, 0.0, 1.0)


            current_step_size = (
                self.step_size_initial
                - (self.step_size_initial - self.step_size_final) * progress
            )
            current_step_size = max(current_step_size, self.step_size_final)
            self.step_size = current_step_size

            compliance_before = self.previous_compliance
            reward_action = 0.0

            # Single cell
            old_density = self.state[cell].item()
            if direction == 'increase':
                new_density = min(old_density + self.step_size, 1.0)
            else:
                new_density = max(old_density - self.step_size, 0.001)

            if new_density != old_density:
                self.state[cell] = new_density
                reward_action += 0.05
            else:
                reward_action -= 0.05

            self.current_step += 1

            main_reward, reward_info = self.calculate_reward() 

            compliance_after = self.current_compliance
            self.previous_compliance = self.current_compliance
            diff = 0
            if compliance_before <  int(1e4):
                diff = compliance_before - compliance_after
                # If diff > 0 => compliance improved
                # If diff < 0 => got worse
                # Scale it so we don't overshadow main_reward
                improvement_scale = 0.05 # or tweak as needed
                reward_improvement = improvement_scale * diff / self.base_compliance[self.beam_type]
            else:
                reward_improvement = 0.0

            # Combine everything
            reward = reward_action + main_reward + reward_improvement

            # Optionally, do mild scaling or clipping
            #reward = reward / (1 + abs(reward)) #np.clip(reward, -5.0, 5.0)
            reward = reward / 5
            #reward = reward / 2.0    #for 6x6 grid works great
            reward = reward / (1 + abs(reward))
            if self.current_compliance < self.best_compliance[self.beam_type] :
                self.best_compliance[self.beam_type] = self.current_compliance
                reward += 0.1  # small bonus for setting a new best
                #done = True

            done = (self.current_step >= self.max_steps)
                #if done:
                    #print(f"here motherfucke is done since {self.current_step}")

            #reward = np.clip( reward, -1.0, 2.0)

            # Info dict
            info = {
                'compliance_before': compliance_before,
                'compliance_after': compliance_after,
                'compliance_diff': diff ,
                'reward_improvement': reward_improvement,
                'reward_action': reward_action,
                'reward_main': main_reward,
                'compliance': self.current_compliance,
                'constraint': self.current_constraint,
                'direction': direction,
                'step': self.current_step,
            }

            return self.state.cpu().numpy(), reward, done, False, info
    

    def calculate_reward(self):
        # Evaluate compliance, constraint, etc.
        with torch.no_grad():
            x = self.state.cpu().numpy()[: self.width * self.height]
            compliance, constraint = fem.optim(args=self.args, x=x)

        self.current_compliance = compliance
        self.current_constraint = constraint

        # If compliance is huge, let's clamp or heavily penalize it
        if compliance > 1e5:
            # or some large negative reward
            return -10.0, {'raw_reward': -100.0, 'compliance': compliance, 'constraint': constraint}


        # Weighted components
        r_compliance = -self.w_compliance * (compliance / self.base_compliance[1])

        densities = self.state[: self.width * self.height].cpu().numpy()


        grey_mask = (densities > 0.2) & (densities < 0.8)
        fraction_grey = np.mean(grey_mask)  # fraction of cells that are "grey"
        r_grey = -1* fraction_grey

        total_mass = np.sum(densities)
        target_total_density = (self.optimal_density + 0.02) * self.width * self.height
        mass_deviation = max(total_mass - target_total_density, 0)
        r_mass = -self.w_total_mass * mass_deviation



        # Sum them, for 6x6 grid its 0.5 + (r_grey + r_mass)
        reward = (
            r_compliance
            + 0.5 * r_grey   # 0.5 * grey for now the best result
            + 0.1 * r_mass      # 0.1 * mass for now the best result

        )


        info = {
            'raw_reward': reward,
            'compliance': compliance,
            'constraint': constraint,
            'r_compliance': r_compliance,
            'r_mass': r_mass,

        }

        return reward, info
 
    #'''
    def render(self, mode='human', step_interval=100):
        """
        Render the current state of the environment.
        """
        if mode == 'human':# and self.current_step % step_interval == 0:
            grid = self.state[:self.width * self.height].cpu().numpy().reshape(self.height, self.width)
            plt.figure(figsize=(6,6))
            plt.imshow(grid, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
            plt.colorbar()
            plt.title(f"Step: {self.current_step} | Reward: {self.reward:.3f} | Compliance: {self.current_compliance:.3f}")
            plt.xlabel("Width")
            plt.ylabel("Height")
            plt.show()
        else:
            pass  # No rendering for other modes or steps

    def close(self):
        """
        Clean up the environment's resources.
        """
        plt.close()
