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
        self.max_steps = self.width * self.height * 10
        self.action_space = gym.spaces.Discrete(2 * self.width * self.height)
        self.state_dim = self.width*self.height  + (self.width + 1)*(self.height + 1) * 2
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.state_dim, ), dtype=np.float32)
        self.step_size_initial = 0.5
        self.step_size_final = 0.05
        self.best_compliance = torch.full((10,), float('inf'), dtype=torch.float32)
        self.beam_type = beam_type
        self.phase_threshold_1 = 0.02
        self.phase_threshold_2 = 0.2
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
            5: self.compute_weight_matrix(size=5),
            3: self.compute_weight_matrix(size=3)
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
        
        temp_state = torch.ones((self.width * self.height), dtype=torch.float32, device=self.device)* 0.5
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
            penalty = 2 * others_sum  # scale as you want
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
        # Map action to cell and direction
        total_cells = self.width * self.height
        if action < 0 or action >= 2 * total_cells:
            raise ValueError(f"Invalid action: {action}")

        cell = action % total_cells  # Cell index
        direction = 'increase' if action >= total_cells else 'decrease'

        # Define density modification parameters
        min_density = 0.001
        max_density = 1.0

        progress = self.current_step / self.max_steps
        progress = np.clip(progress, 0.0, 1.0)

        if progress < self.phase_threshold_1:
            phase = 'phase1'  # 5x5 subgrids
            subgrid_size = 5
        elif progress < self.phase_threshold_2:
            phase = 'phase2'  # 3x3 subgrids
            subgrid_size = 3

        else:
            phase = 'phase3'  # Single cells
            subgrid_size = 1

        # Compute dynamic step size: linearly decay from initial to final step size
        current_step_size = self.step_size_initial - (self.step_size_initial - self.step_size_final) * progress
        current_step_size = max(current_step_size, self.step_size_final)  
        self.step_size = current_step_size
        
        # Initialize reward
        reward = 0.0
        info = {}
        

        if subgrid_size >= 2:
            weights = self.weight_matrices[subgrid_size]  
            cells_to_modify = self.get_subgrid_cells(cell, size=subgrid_size)  
            center = subgrid_size // 2

            for idx, target_cell in enumerate(cells_to_modify):
                # Calculate row and column within the subgrid
                r = idx // subgrid_size
                c = idx % subgrid_size

                weight = weights[r, c].item()  

                adjustment = self.step_size * weight if direction == 'increase' else -self.step_size * weight

                current_density = self.state[target_cell].item()
                new_density = current_density + adjustment

                new_density = np.clip(new_density, min_density, max_density)

                if direction == 'increase' and new_density > current_density:
                    self.state[ target_cell] = new_density
                    reward += 0.05 * weight  
                elif direction == 'decrease' and new_density < current_density:
                    self.state[ target_cell] = new_density
                    reward += 0.05 * weight  
                else:
                    penalty = -0.05 * weight
                    reward += penalty
        
        else:
            current_density = self.state[cell].item()
            if direction == 'increase':
                new_density = min(current_density + self.step_size, max_density)
                if new_density > current_density:
                    self.state[cell] = new_density
                    reward += 0.05  
                else:
                    reward -= 0.1  
            else:  
                new_density = max(current_density - self.step_size, min_density)
                if new_density < current_density:
                    self.state[cell] = new_density
                    reward += 0.05  
                else:
                    reward -= 0.1   

        self.current_step += 1

        # Calculate additional reward components
        additional_reward, reward_info = self.calculate_reward()
        reward += additional_reward
        reward = reward / (abs(reward) + 1)

        relative_threshold_value = self.best_compliance[self.beam_type] * (0.99)
        if self.current_compliance < self.best_compliance[self.beam_type]:
            self.best_compliance[self.beam_type] = self.current_compliance
        if self.current_compliance < relative_threshold_value:
            done = True
            # logging.info(f"Early termination  at step {self.current_step} with compliance {self.current_compliance:.4f}")
            # print(f"Early termination  at step {self.current_step} with compliance {self.current_compliance:.4f}")

        done = self.current_step >= self.max_steps

        info = {
            'compliance': self.current_compliance,
            'constraint': self.current_constraint,
            'modified_cell': cell,
            'direction': direction,
            'new_density': new_density,
            'num_of_changed_cells': self.current_step
        }
        if done:
            logging.info(
                f"Episode done: Step: {self.current_step} | Action: {action} | "
                f"Raw Reward: {reward_info['raw_reward']:.2f} | "
                f"Scaled Reward: {reward_info['scaled_reward']:.4f} | "
                f"Reward Compliance: {reward_info['reward_compliance']:.2f} | "
                f"Reward Density High: {reward_info['reward_density_high']:.2f} | "
                f"Reward Density Low: {reward_info['reward_density_low']:.2f} | "
                f"Reward Total Mass: {reward_info['reward_total_mass']:.2f} | "
                f"Reward Entropy: {reward_info['reward_entropy']:.2f} | "
                f"Total Reward: {reward:.4f} | "
                f"Compliance: {reward_info['compliance']:.2f} | "
                f"Constraint: {reward_info['constraint']:.2f}"
            )
        # print(f"state shaoe before returning it from step function is {self.state.shape}")
        return self.state.cpu().numpy(), reward, done, False, info
  

    def calculate_reward(self):

        with torch.no_grad():
            x = self.state.cpu().numpy()
            x = x[: self.width * self.height]
            compliance, constraint = fem.optim(args=self.args, x=x)
            self.current_compliance = compliance

        penalty_connectivity = self.check_connectivity(x)
        w_compliance = self.w_compliance
        w_density_high = self.w_density_high  
        w_density_low = self.w_density_low
        w_total_mass = self.w_total_mass
        w_entropy = self.w_entropy           

        reward_compliance = -w_compliance * (compliance / 100.0)


        densities = self.state[:self.width * self.height].cpu().numpy()

        density_high_reward = np.mean( (densities - 0.5) ** 2)  
        reward_density_high = w_density_high * density_high_reward

        density_low_reward = np.mean((0.5 - densities) ** 2)  
        reward_density_low = w_density_low * density_low_reward

        target_total_density = self.optimal_density * self.width * self.height
        total_mass = np.sum(densities)
        mass_deviation = max(total_mass - target_total_density, 0)
        reward_total_mass = -w_total_mass * mass_deviation


        #density_entropy = -np.mean(densities * np.log(densities + 1e-8) + 
        #                        (1 - densities) * np.log(1 - densities + 1e-8))
        density_entropy = -np.mean(densities * np.log(np.clip(densities, 1e-8, None)) +
                               (1 - densities) * np.log(np.clip(1 - densities, 1e-8, None)))
        reward_entropy = w_entropy * density_entropy

        reward = (reward_compliance + #reward_connectivity + reward_isolated +
                reward_density_high + reward_density_low + reward_total_mass +
                reward_entropy - penalty_connectivity)
        

        # Large or unbounded rewards can lead to large gradient updates
        scaled_reward =  reward / (1 + abs(reward))

        #scaled_reward = reward
        reward_info = {
            'raw_reward': reward,
            'scaled_reward': scaled_reward,
            'compliance': compliance,
            'constraint': constraint,
            'reward_compliance': reward_compliance,
            'reward_density_high': reward_density_high,
            'reward_density_low': reward_density_low,
            'reward_total_mass': reward_total_mass,
            'reward_entropy': reward_entropy
        }
        # logging.info(f"Constraint: {constraint}, Compliance: {compliance}")

        self.previous_constraint = constraint
        self.current_compliance = compliance
        self.current_constraint = constraint
        self.reward = scaled_reward

        return scaled_reward, reward_info




    def render(self, mode='human', step_interval=100):
        """
        Render the current state of the environment.
        """
        if mode == 'human':# and self.current_step % step_interval == 0:
            grid = self.state[:self.width * self.height].cpu().numpy().reshape(self.height, self.width)
            plt.figure(figsize=(6, 6))
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
