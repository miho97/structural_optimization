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
        self.max_steps = self.width * self.height * 2
        self.action_space = gym.spaces.Discrete(2 * self.width * self.height)
        self.state_dim = self.width*self.height * 2  + (self.width + 1)*(self.height + 1) * 2
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


    def construct_state(self, densities, normals, forces, strain_array):
        # 1. Flatten everything
        #print(type(densities))
        densities_flat = densities.ravel()
        normals_flat   = normals.flatten()
        forces_flat    = forces.ravel()
        strain_flat    = strain_array.flatten()
        #print(f"densities flat are of type {type(densities_flat)}")
        # 2. Normalize strain (using mean-std standardization)
        strain_mean = strain_flat.mean()
        strain_std  = strain_flat.std() + 1e-8  # avoid divide-by-zero
        strain_normalized = (strain_flat - strain_mean) / strain_std

        # 3. Concatenate into a single 1D tensor
        new_state = torch.cat([
            densities_flat.clone().detach(),
            #torch.tensor(densities_flat,      device=self.device, dtype=torch.float32),
            torch.tensor(strain_normalized,   device=self.device, dtype=torch.float32),
            # If you want boundary normals:
            # torch.tensor(normals_flat,      device=self.device, dtype=torch.float32),
            forces_flat.clone().detach()#torch.tensor(forces_flat,         device=self.device, dtype=torch.float32)
        ], dim=0)

        return new_state

    def reset(self,*, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        temp_state = torch.ones((self.width * self.height), dtype=torch.float32, device=self.device)* self.optimal_density
        self.current_step = 0
        self.visited = torch.zeros((self.width * self.height), dtype=bool, device=self.device)
        self.visited_cells = set()
        self.current_compliance = float('inf')
        self.previous_compliance = float('inf')
        self.current_constraint = 1.0
        self.reward = 0
        #self.best_copmliance = float('inf')
        self.current_compliance = float('inf')
        strain = fem.elementwise_strain_energy( x= temp_state.cpu().numpy(), args = self.args)

        self.state = self.construct_state(temp_state, self.normals, self.forces, strain)

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

###################################################
#   standard approach with compliance every step  #
###################################################

    def step(self, action):
        """
        Perform the action and return the new state, reward, done, truncated, and info.
        """
        total_cells = self.width * self.height
        if action < 0 or action >= 2 * total_cells:
            raise ValueError(f"Invalid action: {action}")

        cell = action % total_cells  # which cell
        direction = 'increase' if action >= total_cells else 'decrease'

        # --- 2) Determine subgrid (phase) based on training progress ---
        progress = np.clip(self.current_step / self.max_steps, 0.0, 1.0)
        # For now, subgrid_size remains 1 (but can be changed later)
        #subgrid_size = 1

        subgrid_size = 1
        # --- 3) Step size decay ---
        current_step_size = self.step_size_initial - (self.step_size_initial - self.step_size_final) * progress
        self.step_size = max(current_step_size, self.step_size_final)

        # --- 4) Record compliance BEFORE the action ---
        compliance_before = self.previous_compliance

        # --- 5) Apply the action ---
        reward_action = 0.0
        if subgrid_size >= 2:
            cells_to_modify = self.get_subgrid_cells(cell, size=subgrid_size)
            weights = self.weight_matrices[subgrid_size]
            for idx, target_cell in enumerate(cells_to_modify):
                r = idx // subgrid_size
                c = idx % subgrid_size
                w = weights[r, c].item()
                delta = self.step_size * w if direction == 'increase' else -self.step_size * w
                old_density = self.state[target_cell].item()
                new_density = np.clip(old_density + delta, 0.001, 1.0)
                if new_density != old_density:
                    self.state[target_cell] = new_density
                    reward_action += 0.05 * w
                else:
                    reward_action -= 0.05 * w
        else:
            old_density = self.state[cell].item()
            new_density = (min(old_density + self.step_size, 1.0) if direction == 'increase'
                        else max(old_density - self.step_size, 0.001))
            if new_density != old_density:
                self.state[cell] = new_density
                reward_action += 0.05
            else:
                reward_action -= 0.05

        self.current_step += 1

        # --- 6) Compute domain-specific reward ---
        main_reward, reward_info = self.calculate_reward()  # this updates self.current_compliance

        # --- 7) Compute improvement reward ---
        compliance_after = self.current_compliance
        self.previous_compliance = self.current_compliance
        if compliance_before < 1e4 and compliance_before > 0:
            # Reward is proportional to the relative improvement in compliance.
            improvement = (compliance_before - compliance_after) / (compliance_before + 1e-8)
            r_improve = 0.05 *  improvement
        else:
            r_improve = 0.0

        # --- Combine rewards ---
        # Sum the action reward, domain-specific reward, and improvement reward.
        reward = reward_action + main_reward + r_improve

        # Optionally clip or scale the final reward to keep it in a controlled range.
        reward = np.clip(reward, -1.0, 1.0)

        # Bonus for setting a new best compliance.
        if self.current_compliance < self.best_compliance[self.beam_type]:
            self.best_compliance[self.beam_type] = self.current_compliance
            reward += 0.0

        done = (self.current_step >= self.max_steps)

        info = {
            'compliance_before': compliance_before,
            'compliance_after': compliance_after,
            'compliance_diff': compliance_before - compliance_after,
            'r_improve': r_improve,
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
            x = self.state[: self.width * self.height]
            compliance, constraint = fem.optim(args=self.args, x=x.cpu().numpy())
            strain = fem.elementwise_strain_energy(x=x.cpu().numpy(), args=self.args)
            new_state = self.construct_state(x, self.normals, self.forces, strain)
            self.state = new_state

        self.current_compliance = compliance
        self.current_constraint = constraint

        # If compliance is very high, then we return a heavy penalty.
        if compliance > 1e5:
            return -10.0, {'raw_reward': -10.0, 'compliance': compliance, 'constraint': constraint}

        # --- Component 1: Normalized Compliance Reward ---
        # Measure relative error (if compliance is lower than the baseline, that’s good)
        # Here, lower compliance is desired so we take error = (compliance - baseline) / baseline.
        baseline = self.base_compliance[self.beam_type]
        # print(f"baseline compo+liance is {baseline}")
        compliance_error = (compliance - baseline) / (baseline + 1e-8)  
        # Negative error gives a positive reward (since lower compliance is better)
        r_compliance = -self.w_compliance * compliance_error

        # --- Component 2: Normalized Mass Penalty ---
        densities = self.state[: self.width * self.height].cpu().numpy()
        total_mass = np.sum(densities)
        target_total_mass = self.optimal_density * self.width * self.height
        mass_deviation = (total_mass - target_total_mass) / (target_total_mass + 1e-8)
        r_mass = -self.w_total_mass * mass_deviation

        # --- Component 3: Constraint Penalty (if applicable) ---
        r_constraint = 0.0
        self.constraint_threshold = 0.45
        if constraint > self.constraint_threshold:  # you can define a threshold
            r_constraint = -1* ((constraint - self.constraint_threshold) / self.constraint_threshold)

        # --- Combine Domain-Specific Reward ---
        # You may choose weights here; note that r_compliance is positive when compliance improves.
        domain_reward = r_compliance + 0.5 * r_mass + r_constraint
        domain_reward = r_compliance + r_mass #+ r_constraint
        #domain_reward /= 10 

        # Clip the domain reward to keep it within a modest range.
        domain_reward = np.clip(domain_reward, -1.0, 1.0)

        info = {
            'raw_compliance': compliance,
            'constraint': constraint,
            'r_compliance': r_compliance,
            'r_mass': r_mass,
            'r_constraint': r_constraint,
            'domain_reward': domain_reward,
        }
        return domain_reward, info

#######################
#     Testing         #
#######################
    #'''
    def step(self, action):
            """
            Perform the action and return the new state, reward, done, truncated, and info.
            """
            total_cells = self.width * self.height
            if action < 0 or action >= 2 * total_cells:
                raise ValueError(f"Invalid action: {action}")

            cell = action % total_cells  # which cell
            direction = 'increase' if action >= total_cells else 'decrease'

            # --- 2) Determine subgrid (phase) based on training progress ---
            progress = np.clip(self.current_step / self.max_steps, 0.0, 1.0)
            # For now, subgrid_size remains 1 (but can be changed later)
            #subgrid_size = 1

            subgrid_size = 1
            # --- 3) Step size decay ---
            current_step_size = self.step_size_initial - (self.step_size_initial - self.step_size_final) * progress
            self.step_size = max(current_step_size, self.step_size_final)

            # --- 4) Record compliance BEFORE the action ---
            compliance_before = self.previous_compliance

            # --- 5) Apply the action ---
            reward_action = 0.0
            if subgrid_size >= 2:
                cells_to_modify = self.get_subgrid_cells(cell, size=subgrid_size)
                weights = self.weight_matrices[subgrid_size]
                for idx, target_cell in enumerate(cells_to_modify):
                    r = idx // subgrid_size
                    c = idx % subgrid_size
                    w = weights[r, c].item()
                    delta = self.step_size * w if direction == 'increase' else -self.step_size * w
                    old_density = self.state[target_cell].item()
                    new_density = np.clip(old_density + delta, 0.001, 1.0)
                    if new_density != old_density:
                        self.state[target_cell] = new_density
                        reward_action += 0.05 * w
                    else:
                        reward_action -= 0.05 * w
            else:
                old_density = self.state[cell].item()
                new_density = (min(old_density + self.step_size, 1.0) if direction == 'increase'
                            else max(old_density - self.step_size, 0.001))
                if new_density != old_density:
                    self.state[cell] = new_density
                    reward_action += 0.00
                else:
                    reward_action -= 0.00

            self.current_step += 1

            # --- 6) Compute domain-specific reward ---
            main_reward, reward_info = self.calculate_reward()  # this updates self.current_compliance

            # --- 7) Compute improvement reward ---
            compliance_after = self.current_compliance
            self.previous_compliance = self.current_compliance
            if compliance_before < 1e4 and compliance_before > 0:
                # Reward is proportional to the relative improvement in compliance.
                improvement = (compliance_before - compliance_after) / (compliance_before + 1e-8)
                r_improve = 0.00 *  improvement
            else:
                r_improve = 0.0

            # --- Combine rewards ---
            # Sum the action reward, domain-specific reward, and improvement reward.
            reward = reward_action + main_reward + r_improve

            # Optionally clip or scale the final reward to keep it in a controlled range.
            reward = np.clip(reward, -1.0, 1.0)
            #reward = reward / ( abs(reward) + 1)

            # Bonus for setting a new best compliance.
            if self.current_compliance < self.best_compliance[self.beam_type]:
                self.best_compliance[self.beam_type] = self.current_compliance
                reward += 0.0

            done = (self.current_step >= self.max_steps)

            info = {
                'compliance_before': compliance_before,
                'compliance_after': compliance_after,
                'compliance_diff': compliance_before - compliance_after,
                'r_improve': r_improve,
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
        with torch.no_grad():
            x = self.state[: self.width * self.height]
            compliance, constraint = fem.optim(args=self.args, x=x.cpu().numpy())
            strain = fem.elementwise_strain_energy(x=x.cpu().numpy(), args=self.args)
            new_state = self.construct_state(x, self.normals, self.forces, strain)
            self.state = new_state

        self.current_compliance = compliance
        self.current_constraint = constraint

        if compliance > 1e5:
            return -10.0, {'raw_reward': -10.0, 'compliance': compliance, 'constraint': constraint}

        # Use an adaptive or chosen baseline.
        baseline = self.base_compliance[self.beam_type]/5  # might be 1180
        # Compute the ratio and take logarithm.
        # When compliance == baseline, log(1) = 0. For compliance below baseline, log(<1) < 0.
        ratio = compliance / (baseline + 1e-8)
        r_compliance = -self.w_compliance * np.log(ratio)  # This becomes positive when compliance < baseline

        densities = self.state[: self.width * self.height].cpu().numpy()
        total_mass = np.sum(densities)
        target_total_mass = self.optimal_density * self.width * self.height
        mass_deviation = (total_mass - target_total_mass) / (target_total_mass + 1e-8)
        r_mass = -self.w_total_mass * mass_deviation

        r_constraint = 0.0
        self.constraint_threshold = 0.45
        if constraint > self.constraint_threshold:
            r_constraint = -1 * ((constraint - self.constraint_threshold) / self.constraint_threshold)

        domain_reward = r_compliance + r_mass #+ r_constraint
        domain_reward = np.clip(domain_reward, -1.0, 1.0)

        info = {
            'raw_compliance': compliance,
            'constraint': constraint,
            'r_compliance': r_compliance,
            'r_mass': r_mass,
            'r_constraint': r_constraint,
            'domain_reward': domain_reward,
        }
        return domain_reward, info
    #'''
####################################
#     Trying sparse reward         #
####################################
    '''
    def step(self, action):
        """
        Perform the action and return the new state, reward, done, truncated, and info.
        In this version, the immediate reward (from the action) is very small,
        and an aggregated reward is computed every 10% of the episode.
        """
        total_cells = self.width * self.height
        if action < 0 or action >= 2 * total_cells:
            raise ValueError(f"Invalid action: {action}")

        cell = action % total_cells  # which cell
        direction = 'increase' if action >= total_cells else 'decrease'

        # --- 1) Determine progress and step size decay ---
        progress = np.clip(self.current_step / self.max_steps, 0.0, 1.0)
        subgrid_size = 1  # we use a single-cell update for now
        current_step_size = self.step_size_initial - (self.step_size_initial - self.step_size_final) * progress
        self.step_size = max(current_step_size, self.step_size_final)

        # --- 2) Record compliance BEFORE the action (for aggregation) ---
        # At the start of the episode, initialize previous_compliance.
        if self.current_step == 0:
            # You can choose an initial baseline; here we use self.base_compliance[1] as an example.
            self.previous_compliance = self.base_compliance[1]
        compliance_before = self.previous_compliance

        # --- 3) Apply the action ---
        reward_action = 0.0
        if subgrid_size >= 2:
            cells_to_modify = self.get_subgrid_cells(cell, size=subgrid_size)
            weights = self.weight_matrices[subgrid_size]
            for idx, target_cell in enumerate(cells_to_modify):
                r = idx // subgrid_size
                c = idx % subgrid_size
                w = weights[r, c].item()
                delta = self.step_size * w if direction == 'increase' else -self.step_size * w
                old_density = self.state[target_cell].item()
                new_density = np.clip(old_density + delta, 0.001, 1.0)
                if new_density != old_density:
                    self.state[target_cell] = new_density
                    reward_action += 0.05 * w  # small local reward if desired
                else:
                    reward_action -= 0.05 * w
        else:
            old_density = self.state[cell].item()
            new_density = (min(old_density + self.step_size, 1.0) if direction == 'increase'
                        else max(old_density - self.step_size, 0.001))
            if new_density != old_density:
                self.state[cell] = new_density
                reward_action += 0.01  # you may set a small value here (or 0)
            else:
                reward_action -= 0.01

        self.current_step += 1

        # --- 4) Immediate reward (local signal) ---
        immediate_reward = reward_action

        # --- 5) Periodically aggregate domain reward every 10% of the episode ---
        segment_reward = 0.0
        segment_info = {}
        segment_interval = int(self.max_steps * 0.1)  # every 10% of episode length
        # Check if we are at a segment boundary or at the end.
        if (self.current_step % segment_interval == 0) or (self.current_step >= self.max_steps):
            segment_reward, segment_info = self.calculate_segment_reward(compliance_before)
            # Use different multipliers for intermediate and terminal segments.
            terminal_multiplier = 2.0
            intermediate_multiplier = 0.5
            if self.current_step >= self.max_steps:
                segment_reward *= terminal_multiplier
            else:
                segment_reward *= intermediate_multiplier
            # Update baseline for the next segment.
            self.previous_compliance = self.current_compliance

        # --- 6) Combine rewards ---
        total_reward = immediate_reward + segment_reward
        total_reward = np.clip(total_reward, -1.0, 1.0)

        done = (self.current_step >= self.max_steps)

        # --- 7) Package diagnostic info ---
        info = {
            'compliance_before': compliance_before,
            'compliance_after': self.current_compliance,
            'compliance_diff': compliance_before - self.current_compliance,
            'segment_reward': segment_reward,
            'segment_info': segment_info,
            'immediate_reward': immediate_reward,
            'total_reward': total_reward,
            'step': self.current_step,
            'compliance': self.current_compliance,
            'constraint': self.current_constraint,
        }
        return self.state.cpu().numpy(), total_reward, done, False, info


    def calculate_segment_reward(self, compliance_initial):
        """
        This function aggregates the domain-specific metrics over a segment (e.g., 10% of the episode).
        It computes an aggregated reward based on the current design metrics, comparing the current
        compliance to a baseline.
        
        We use a logarithmic transformation on the compliance ratio, plus a mass penalty.
        """
        with torch.no_grad():
            x = self.state[: self.width * self.height]
            # Recompute design metrics (this updates self.state as well)
            compliance, constraint = fem.optim(args=self.args, x=x.cpu().numpy())
            strain = fem.elementwise_strain_energy(x=x.cpu().numpy(), args=self.args)
            new_state = self.construct_state(x, self.normals, self.forces, strain)
            self.state = new_state

        self.current_compliance = compliance
        self.current_constraint = constraint

        # If compliance is extremely high, return a heavy penalty.
        if compliance > 1e5:
            return -10.0, {'raw_reward': -10.0, 'compliance': compliance, 'constraint': constraint}

        # --- Aggregated Compliance Reward ---
        # Use the baseline (for instance, initial design value, e.g., 1180)
        baseline = self.base_compliance[self.beam_type] / 10 # e.g., 1180
        ratio = compliance / (baseline + 1e-8)
        r_compliance = -self.w_compliance * np.log(ratio)
        # When compliance < baseline, log(ratio) < 0 so r_compliance > 0

        # --- Aggregated Mass Penalty ---
        densities = self.state[: self.width * self.height].cpu().numpy()
        total_mass = np.sum(densities)
        target_total_mass = self.optimal_density * self.width * self.height
        mass_deviation = (total_mass - target_total_mass) / (target_total_mass + 1e-8)
        r_mass = -self.w_total_mass * mass_deviation

        # --- Optional Constraint Penalty ---
        r_constraint = 0.0
        self.constraint_threshold = 0.45
        if constraint > self.constraint_threshold:
            r_constraint = -1 * ((constraint - self.constraint_threshold) / self.constraint_threshold)

        # Combine aggregated rewards.
        aggregated_reward = r_compliance + 2 * r_mass + r_constraint
        aggregated_reward = np.clip(aggregated_reward, -1.0, 1.0)

        info = {
            'raw_compliance': compliance,
            'constraint': constraint,
            'r_compliance': r_compliance,
            'r_mass': r_mass,
            'r_constraint': r_constraint,
            'aggregated_reward': aggregated_reward,
            'improvement_ratio': (compliance_initial - compliance) / (compliance_initial + 1e-8)
        }
        return aggregated_reward, info

#############################################################
#  less sparse approach but solver stil called every step   #
#############################################################
    #'''
    '''
    def step(self, action):
        """
        Perform the action and return the new state, reward, done, truncated, and info.
        In this version, the immediate reward (from the action) is very small.
        We aggregate a reward every 10% of the episode based on the improvement in compliance,
        and also provide a smaller continuous bonus every 5% of the episode.
        """
        total_cells = self.width * self.height
        if action < 0 or action >= 2 * total_cells:
            raise ValueError(f"Invalid action: {action}")

        cell = action % total_cells  # which cell
        direction = 'increase' if action >= total_cells else 'decrease'

        # --- 1) Determine progress and step size decay ---
        progress = np.clip(self.current_step / self.max_steps, 0.0, 1.0)
        subgrid_size = 1  # single-cell update for now
        current_step_size = self.step_size_initial - (self.step_size_initial - self.step_size_final) * progress
        self.step_size = max(current_step_size, self.step_size_final)

        # --- 2) Record compliance baseline for the segment ---
        # At the very beginning of the episode, also set the segment baseline.
        if self.current_step == 0:
            self.previous_compliance = self.base_compliance[1]
            self.segment_baseline_compliance = self.previous_compliance
        compliance_before = self.segment_baseline_compliance

        # --- 3) Apply the action ---
        reward_action = 0.0
        if subgrid_size >= 2:
            cells_to_modify = self.get_subgrid_cells(cell, size=subgrid_size)
            weights = self.weight_matrices[subgrid_size]
            for idx, target_cell in enumerate(cells_to_modify):
                r = idx // subgrid_size
                c = idx % subgrid_size
                w = weights[r, c].item()
                delta = self.step_size * w if direction == 'increase' else -self.step_size * w
                old_density = self.state[target_cell].item()
                new_density = np.clip(old_density + delta, 0.001, 1.0)
                if new_density != old_density:
                    self.state[target_cell] = new_density
                    reward_action += 0.05 * w
                else:
                    reward_action -= 0.05 * w
        else:
            old_density = self.state[cell].item()
            new_density = (min(old_density + self.step_size, 1.0) if direction == 'increase'
                        else max(old_density - self.step_size, 0.001))
            if new_density != old_density:
                self.state[cell] = new_density
                reward_action += 0.01  # small immediate reward
            else:
                reward_action -= 0.01

        self.current_step += 1

        # --- 4) Immediate reward (local signal) ---
        immediate_reward = reward_action

        # --- 5) Continuous immediate bonus every 5% of episode ---
        bonus_reward = 0.0
        bonus_interval = int(self.max_steps * 0.01)  # every 5% of the episode
        #if self.current_step % 2 == 0:
        #    with torch.no_grad():
        ##        x = self.state[: self.width * self.height]
        #        compliance_current, _ = fem.optim(args=self.args, x=x.cpu().numpy())
            # Compute improvement relative to the segment baseline.
        #    improvement = (self.segment_baseline_compliance - compliance_current) / (self.segment_baseline_compliance + 1e-8)
        #    alpha_immediate = 0.2  # you can adjust this multiplier
        #    bonus_reward = alpha_immediate * improvement

        # --- 6) Periodically aggregate domain reward every 10% of the episode ---
        segment_reward = 0.0
        segment_info = {}
        segment_interval = int(self.max_steps * 0.01)  # every 10% of episode length
        segment_interval = 2
        if (self.current_step % segment_interval == 0) or (self.current_step >= self.max_steps):
            segment_reward, segment_info = self.calculate_segment_reward(self.segment_baseline_compliance)
            # Use a higher multiplier for the terminal segment.
            terminal_multiplier = 10.0
            intermediate_multiplier = 1.0
            if self.current_step >= self.max_steps:
                segment_reward *= terminal_multiplier
            else:
                segment_reward *= intermediate_multiplier
            # Update the segment baseline for the next segment.
            self.segment_baseline_compliance = self.current_compliance

        # --- 7) Combine rewards ---
        total_reward = immediate_reward + bonus_reward + segment_reward
        total_reward = np.clip(total_reward, -1.0, 1.0)
        done = (self.current_step >= self.max_steps)

        info = {
            'compliance_baseline': compliance_before,
            'compliance_current': self.current_compliance,
            'compliance_improvement': compliance_before - self.current_compliance,
            'immediate_reward': immediate_reward,
            'bonus_reward': bonus_reward,
            'segment_reward': segment_reward,
            'segment_info': segment_info,
            'total_reward': total_reward,
            'step': self.current_step,
            'constraint': self.current_constraint,
            'compliance': self.current_compliance
        }
        return self.state.cpu().numpy(), total_reward, done, False, info


    def calculate_segment_reward(self, compliance_baseline):
        """
        Aggregates domain-specific metrics over the current segment (e.g., 10% of the episode).
        Computes an aggregated reward based on the current design metrics (compliance, mass, etc.)
        compared to the segment baseline.
        """
        with torch.no_grad():
            x = self.state[: self.width * self.height]
            # Recompute design metrics (this call updates self.state)
            compliance, constraint = fem.optim(args=self.args, x=x.cpu().numpy())
            strain = fem.elementwise_strain_energy(x=x.cpu().numpy(), args=self.args)
            new_state = self.construct_state(x, self.normals, self.forces, strain)
            self.state = new_state

        self.current_compliance = compliance
        self.current_constraint = constraint

        # If compliance is extremely high, return a heavy penalty.
        if compliance > 1e5:
            return -10.0, {'raw_reward': -10.0, 'compliance': compliance, 'constraint': constraint}

        # --- Aggregated Compliance Reward ---
        # Use the baseline passed from the segment (compliance_baseline)
        ratio = compliance / (compliance_baseline + 1e-8)
        r_compliance = -self.w_compliance * np.log(ratio)
        # When compliance < baseline, np.log(ratio) < 0, so r_compliance becomes positive.

        # --- Aggregated Mass Penalty ---
        densities = self.state[: self.width * self.height].cpu().numpy()
        total_mass = np.sum(densities)
        target_total_mass = self.optimal_density * self.width * self.height
        mass_deviation = (total_mass - target_total_mass) / (target_total_mass + 1e-8)
        r_mass = -self.w_total_mass * mass_deviation

        # --- Optional Constraint Penalty ---
        r_constraint = 0.0
        self.constraint_threshold = 0.45
        if constraint > self.constraint_threshold:
            r_constraint = -1 * ((constraint - self.constraint_threshold) / self.constraint_threshold)

        aggregated_reward = r_compliance + 0.1 * r_mass #+ r_constraint
        aggregated_reward = np.clip(aggregated_reward, -1.0, 1.0)

        info = {
            'raw_compliance': compliance,
            'constraint': constraint,
            'r_compliance': r_compliance,
            'r_mass': r_mass,
            'r_constraint': r_constraint,
            'aggregated_reward': aggregated_reward,
            'improvement_ratio': (compliance_baseline - compliance) / (compliance_baseline + 1e-8)
        }
        return aggregated_reward, info
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
