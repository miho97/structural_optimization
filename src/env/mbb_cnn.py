import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  
from utils import fem
import logging
import torch.nn.functional as F

class BeamOptimizationEnv(gym.Env):
    metadata = {'render.modes': ['human']}  
    
    def __init__(self, width=4, height=4, density=0.4, step_size=0.5, 
                 optimal_density=0.5, reward_weights=None, beam_type=1):
        super(BeamOptimizationEnv, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.width = width
        self.height = height
        self.density = density
        self.step_size_initial = 1.0
        self.step_size_final = 0.05
        self.optimal_density = optimal_density  
        self.max_steps = self.width * self.height * 2
        self.action_space = gym.spaces.Discrete(2 * self.width * self.height)
        
        # Determine number of edges based on beam structure
        self.number_of_edges = (self.width * (self.height + 1)) + ((self.width + 1) * self.height)
        self.state_dim = (self.width * self.height) + (self.number_of_edges * 2)  # Densities + normals + forces
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5, width+1, height+1),
            dtype=np.float32
        )
        
        self.best_compliance = torch.full((10,), float('inf'), dtype=torch.float32, device=self.device)
        self.beam_type = beam_type
        self.phase_threshold_1 = 0.05
        self.phase_threshold_2 = 0.1

        # Initialize beam properties
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

        normals, forces, _ = beam_functions[beam_type](width, height, density)
        self.normals = torch.tensor(normals, dtype=torch.float32, device=self.device)
        self.forces = torch.tensor(forces, dtype=torch.float32, device=self.device)
        self.args = fem.get_args(self.normals.cpu().numpy(), self.forces.cpu().numpy(), density)
        
        self.current_compliance = float('inf')
        self.previous_compliance = float('inf')
        self.current_constraint = 0.5 
        self.reward = 0
        
        # Set reward weights
        if reward_weights is None:
            self.w_compliance = 1.0
            self.w_density_high = 1.0   
            self.w_density_low = 1.0    
            self.w_total_mass = 1.0
            self.w_entropy = 0.5
        else:
            self.w_compliance = reward_weights.get('w_compliance', 1.0)
            self.w_density_high = reward_weights.get('w_density_high', 1.0)  
            self.w_density_low = reward_weights.get('w_density_low', 1.0)    
            self.w_total_mass = reward_weights.get('w_total_mass', 1.0)
            self.w_entropy = reward_weights.get('w_entropy', 0.5)

        self.weight_matrices = {
            7: self.compute_weight_matrix(size=7),
            5: self.compute_weight_matrix(size=5),
            3: self.compute_weight_matrix(size=3)
        }

        self.base_compliance = {
            beam_type: fem.optim(fem.get_args(*beam_func()), x=None)[0]
            for beam_type, beam_func in beam_functions.items()
        }

        self.reset()

    def construct_cnn_state(self, densities, normals, forces):
        """
        Constructs a multi-channel state tensor for CNN input.
        """
        # Ensure densities is 2D: [width, height]
        if densities.dim() == 1:
            densities = densities.view(self.width, self.height)
        
        # Pad densities to (width+1, height+1)
        densities_padded = F.pad(densities, (0, 1, 0, 1), mode='constant', value=0.0)
        
        # Reshape normals and forces
        normals_grid = normals.view(self.width + 1, self.height + 1, 2)
        forces_grid = forces.view(self.width + 1, self.height + 1, 2)
        
        # Split into separate channels
        normals_x = normals_grid[:, :, 0]
        normals_y = normals_grid[:, :, 1]
        forces_x = forces_grid[:, :, 0]
        forces_y = forces_grid[:, :, 1]
        
        # Stack into channels
        state_channels = torch.stack([
            densities_padded,  # Channel 1: Densities
            normals_x,        # Channel 2: Normals X
            normals_y,        # Channel 3: Normals Y
            forces_x,         # Channel 4: Forces X
            forces_y          # Channel 5: Forces Y
        ], dim=0)  # Shape: (5, width+1, height+1)
        
        return state_channels


    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Initialize densities as a 2D tensor: [width, height]
        densities = torch.ones((self.width, self.height), dtype=torch.float32, device=self.device) * self.optimal_density
        
        # Reset other variables
        self.current_step = 0
        self.visited = torch.zeros((self.width * self.height), dtype=torch.bool, device=self.device)
        self.visited_cells = set()
        self.current_compliance = float('inf')
        self.previous_compliance = float('inf')
        self.current_constraint = 1.0
        self.reward = 0
        
        # Construct the initial state
        self.state = self.construct_cnn_state(densities, self.normals, self.forces)
        
        return self.state.cpu().numpy(), {}


    def check_connectivity(self, densities):
        """
        Checks the connectivity of the structure based on current densities.
        Returns a penalty based on the number and size of disconnected components.
        """
        grid = densities.reshape(self.height, self.width)  # Shape: (height, width)
        binary_mask = (grid > 0.2).astype(np.uint8)
        visited = np.zeros_like(binary_mask, dtype=bool)

        def neighbors(r, c):
            for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    yield nr, nc

        components = []
        for r in range(self.height):
            for c in range(self.width):
                if binary_mask[r, c] == 1 and not visited[r, c]:
                    queue = [(r, c)]
                    visited[r, c] = True
                    size = 0
                    while queue:
                        rr, cc = queue.pop()
                        size += 1
                        for rr2, cc2 in neighbors(rr, cc):
                            if binary_mask[rr2, cc2] == 1 and not visited[rr2, cc2]:
                                visited[rr2, cc2] = True
                                queue.append((rr2, cc2))
                    components.append(size)

        if len(components) <= 1:
            return 0.0  # No penalty
        else:
            largest = max(components)
            others_sum = sum(components) - largest
            penalty = 0.1 * others_sum  # Scale as needed
            return penalty

    def calculate_reward(self):
        """
        Calculate the reward based on compliance and constraints.
        """
        # Extract densities and forces from the state
        densities_padded = self.state[0, :, :].cpu().numpy()  # Shape: (width+1, height+1)
        normals_x = self.state[1, :, :].cpu().numpy()
        normals_y = self.state[2, :, :].cpu().numpy()
        forces_x = self.state[3, :, :].cpu().numpy()
        forces_y = self.state[4, :, :].cpu().numpy()

        # Flatten densities back to original grid size if necessary
        # Assuming padding was added, remove it
        densities = densities_padded[:self.width, :self.height]  # Shape: (width, height)
        
        # Flatten forces
        forces = np.stack([forces_x, forces_y], axis=-1).reshape(-1, 2)  # Shape: (number_of_edges, 2)
        
        # Compute compliance and constraint
        compliance, constraint = fem.optim(args=self.args, x=densities)
        
        self.current_compliance = compliance
        self.current_constraint = constraint
        
        # Heavy penalty if compliance is excessively large
        if compliance > 1e5:
            return -10.0, {'raw_reward': -100.0, 'compliance': compliance, 'constraint': constraint}
        
        # Penalty for connectivity issues
        penalty_connectivity = self.check_connectivity(densities)
        
        # Weighted components
        r_compliance = -self.w_compliance * (compliance / self.base_compliance[self.beam_type])
        
        # Grey mask for densities between 0.2 and 0.8
        grey_mask = (densities > 0.2) & (densities < 0.8)
        fraction_grey = np.mean(grey_mask)  # Fraction of cells that are "grey"
        
        r_grey = -1 * fraction_grey
        
        # Density low reward
        density_low_reward = np.mean((0.5 - densities) ** 2)
        r_density_low = self.w_density_low * density_low_reward
        
        # Mass penalty or deviation
        total_mass = np.sum(densities)
        target_total_density = self.optimal_density * self.width * self.height
        mass_deviation = max(total_mass - target_total_density, 0)
        r_mass = -self.w_total_mass * mass_deviation
        
        # Entropy
        density_entropy = -np.mean(
            densities * np.log(np.clip(densities, 1e-8, None)) +
            (1 - densities) * np.log(np.clip(1 - densities, 1e-8, None))
        )
        r_entropy = self.w_entropy * density_entropy
        
        # Connectivity penalty
        r_connectivity = -penalty_connectivity
        
        # Sum them
        reward = (
            r_compliance
            + r_grey
            + r_density_low
            + r_mass
            # + r_entropy
            # + r_connectivity
        )
        
        info = {
            'raw_reward': reward,
            'compliance': compliance,
            'constraint': constraint,
            'r_compliance': r_compliance,
            'r_density_low': r_density_low,
            'r_mass': r_mass,
            'r_entropy': r_entropy,
            'r_connectivity': r_connectivity
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

        cell = action % total_cells  # Which cell
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
        compliance_before = self.previous_compliance

        # --- 5) Apply the action (coarse or single-cell) ---
        reward_action = 0.0
        if subgrid_size >= 2:
            # Modify subgrid
            cells_to_modify = self.get_subgrid_cells(cell, size=subgrid_size)
            weights = self.weight_matrices[subgrid_size]  # e.g., shape (5,5) or (3,3)

            for idx, target_cell in enumerate(cells_to_modify):
                r = idx // subgrid_size
                c = idx % subgrid_size
                w = weights[r, c].item()

                # Positive or negative update
                delta = self.step_size * w if direction == 'increase' else -self.step_size * w

                old_density = self.state[0, target_cell // (self.height + 1), target_cell % (self.height + 1)].item()
                new_density = np.clip(old_density + delta, 0.001, 1.0)

                if new_density != old_density:
                    # Small reward for successfully changing density
                    self.state[0, target_cell // (self.height + 1), target_cell % (self.height + 1)] = new_density
                    reward_action += 0.05 * w
                else:
                    # Penalty if we hit a boundary
                    reward_action -= 0.05 * w
        else:
            # Single cell
            row = cell // self.width
            col = cell % self.width
            padded_row = row
            padded_col = col
            old_density = self.state[0, padded_row, padded_col].item()
            if direction == 'increase':
                new_density = min(old_density + self.step_size, 1.0)
            else:
                new_density = max(old_density - self.step_size, 0.001)

            if new_density != old_density:
                self.state[0, padded_row, padded_col] = new_density
                reward_action += 0.05
            else:
                reward_action -= 0.05

        self.current_step += 1

        # --- 6) Compute domain-specific reward (compliance, mass, etc.) ---
        main_reward, reward_info = self.calculate_reward() 
        # This sets self.current_compliance internally, so now we have the AFTER compliance.

        # --- 7) Intermediate compliance-improvement reward ---
        compliance_after = self.current_compliance
        self.previous_compliance = self.current_compliance
        diff = 0
        if compliance_before < 1e4:
            diff = compliance_before - compliance_after
            # If diff > 0 => compliance improved
            # If diff < 0 => got worse
            # Scale it so we don't overshadow main_reward
            improvement_scale = 0.05  # or tweak as needed
            reward_improvement = improvement_scale * diff / self.base_compliance[self.beam_type]
        else:
            reward_improvement = 0.0

        # Combine everything
        reward = reward_action + main_reward + reward_improvement

        # Optionally, do mild scaling or clipping
        reward = reward / (1 + abs(reward))  # np.clip(reward, -5.0, 5.0)

        if self.current_compliance < self.best_compliance[self.beam_type]:
            self.best_compliance[self.beam_type] = self.current_compliance
            reward += 0.1  # Small bonus for setting a new best

        done = (self.current_step >= self.max_steps)

        # Info dict
        info = {
            'compliance_before': compliance_before,
            'compliance_after': compliance_after,
            'compliance_diff': diff,
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
        """
        Calculate the reward based on compliance and constraints.
        """
        # Extract densities and forces from the state
        densities_padded = self.state[0, :, :].cpu().numpy()  # Shape: (width+1, height+1)
        normals_x = self.state[1, :, :].cpu().numpy()
        normals_y = self.state[2, :, :].cpu().numpy()
        forces_x = self.state[3, :, :].cpu().numpy()
        forces_y = self.state[4, :, :].cpu().numpy()

        # Flatten densities back to original grid size
        densities = densities_padded[:self.width, :self.height]  # Shape: (width, height)
        
        # Flatten forces
        forces = np.stack([forces_x, forces_y], axis=-1).reshape(-1, 2)  # Shape: (number_of_edges, 2)
        
        # Compute compliance and constraint
        compliance, constraint = fem.optim(args=self.args, x=densities)
        
        self.current_compliance = compliance
        self.current_constraint = constraint
        
        # Heavy penalty if compliance is excessively large
        if compliance > 1e5:
            return -10.0, {'raw_reward': -100.0, 'compliance': compliance, 'constraint': constraint}
        
        # Penalty for connectivity issues
        penalty_connectivity = self.check_connectivity(densities)
        
        # Weighted components
        r_compliance = -self.w_compliance * (compliance / self.base_compliance[self.beam_type])
        
        # Grey mask for densities between 0.2 and 0.8
        grey_mask = (densities > 0.2) & (densities < 0.8)
        fraction_grey = np.mean(grey_mask)  # Fraction of cells that are "grey"
        
        r_grey = -1 * fraction_grey
        
        # Density low reward
        density_low_reward = np.mean((0.5 - densities) ** 2)
        r_density_low = self.w_density_low * density_low_reward
        
        # Mass penalty or deviation
        total_mass = np.sum(densities)
        target_total_density = self.optimal_density * self.width * self.height
        mass_deviation = max(total_mass - target_total_density, 0)
        r_mass = -self.w_total_mass * mass_deviation
        
        # Entropy
        density_entropy = -np.mean(
            densities * np.log(np.clip(densities, 1e-8, None)) +
            (1 - densities) * np.log(np.clip(1 - densities, 1e-8, None))
        )
        r_entropy = self.w_entropy * density_entropy
        
        # Connectivity penalty
        r_connectivity = -penalty_connectivity
        
        # Sum them
        reward = (
            r_compliance
            + r_grey
            + r_density_low
            + r_mass
            # + r_entropy
            # + r_connectivity
        )
        
        info = {
            'raw_reward': reward,
            'compliance': compliance,
            'constraint': constraint,
            'r_compliance': r_compliance,
            'r_density_low': r_density_low,
            'r_mass': r_mass,
            'r_entropy': r_entropy,
            'r_connectivity': r_connectivity
        }
        
        return reward, info

    def render(self, mode='human', step_interval=100):
        """
        Render the current state of the environment.
        """
        if mode == 'human':
            densities_padded = self.state[0, :, :].cpu().numpy()
            grid = densities_padded[:self.width, :self.height].reshape(self.width, self.height)
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
        for r in range(size):
            for c in range(size):
                distance = max(abs(r - center), abs(c - center))
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
        weights /= weights.max()
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def get_subgrid_cells(self, cell, size):
        """
        Retrieves cells within a subgrid centered around a specific cell.
        
        Args:
            cell (int): Central cell index.
            size (int): Size of the subgrid.
        
        Returns:
            list: List of cell indices within the subgrid.
        """
        subgrid_cells = []
        half_size = size // 2
        row = cell // self.width
        col = cell % self.width

        for dr in range(-half_size, half_size + 1):
            for dc in range(-half_size, half_size + 1):
                new_row = row + dr
                new_col = col + dc
                if 0 <= new_row < self.width + 1 and 0 <= new_col < self.height + 1:
                    neighbor = new_row * (self.height + 1) + new_col
                    subgrid_cells.append(neighbor)

        return subgrid_cells
