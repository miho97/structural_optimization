import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  
from utils import fem
import logging

class BeamOptimizationEnv(gym.Env):
    metadata = {'render.modes': ['human']}  
    def __init__(self, width=4, height=4, density=0.4, step_size=0.05, optimal_density=0.5, reward_weights = None, beam_type=1 ):
        super(BeamOptimizationEnv, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.width = width
        self.height = height
        self.density = density
        self.step_size = step_size  
        self.optimal_density = optimal_density  
        self.max_steps = self.width * self.height * 10
        self.action_space = gym.spaces.Discrete(2 * self.width * self.height)  
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.width * self.height,), dtype=np.float32)
        self.beam_type = beam_type
        # self.reset()

        beam_functions = {
            1: fem.mbb_beam_1,
            2: fem.mbb_beam_2,
            3: fem.mbb_beam_3,
            4: fem.mbb_beam_4
        }
        
        # normals, forces, _ = fem.mbb_beam(width, height, density)
        if self.beam_type > 0:
            normals, forces, _ = beam_functions[beam_type](width, height, density)
        else:
            normals, forces, _ = beam_functions[1](width, height, density)

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

    def reset(self,*, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.state = torch.ones((self.width * self.height), dtype=torch.float32, device=self.device)* 0.5
        self.current_step = 0
        self.visited = torch.zeros((self.width * self.height), dtype=bool, device=self.device)
        self.visited_cells = set()
        self.current_compliance = float('inf')
        self.previous_compliance = float('inf')
        self.current_constraint = 1.0
        self.reward = 0

        if self.beam_type == 0:
            self.forces = self.randomize_forces()
            #self.normals = self.randomize_supports()
            #self.normals += self.randomize_normals()
            self.args = fem.get_args(self.normals.cpu().numpy(), self.forces.cpu().numpy(), self.density)

        return self.state.cpu().numpy(),{}
    
    def randomize_supports(self):
 
        supports = np.zeros((self.width + 1, self.height + 1, 2), dtype=np.float32)
        

        supports[-1, -1, 1] = 1
        supports[0, self.height, 0] = 1

        return torch.tensor(supports, dtype=torch.float32, device=self.device)

    def randomize_forces(self):

        num_forces = np.random.choice([1, 2, 3])
        forces = np.zeros((self.width + 1, self.height + 1, 2), dtype=np.float32)  
        total_positions = (self.width + 1) * (self.height + 1)
        
        selected_indices = np.random.choice(total_positions, size=num_forces, replace=False)
        
        for idx in selected_indices:
            x, y = divmod(idx, self.height + 1)
            axis = np.random.choice(['x', 'y'])
            direction = np.random.choice([1, -1])
            if axis == 'x':
                forces[x, y, 0] = direction * 1.0  
            else:
                forces[x, y, 1] = direction * 1.0  
        
        return torch.tensor(forces, dtype=torch.float32, device=self.device)
    
    def randomize_forces(self):

        num_forces = np.random.choice([1,2,3])
        forces = np.zeros((self.width + 1, self.height + 1, 2), dtype=np.float32)  
        selected_indices = np.random.choice([0,1,2,3,4], size=num_forces, replace=False)
        
        for idx in selected_indices:
            axis = np.random.choice(['x','y'])
            direction = np.random.choice([1, -1])
            if axis == 'x':
                forces[0, idx, 0] = direction * 1.0  
            else:
                forces[0, idx, 1] = direction * 1.0
        return torch.tensor(forces, dtype=torch.float32, device=self.device)

    def randomize_normals(self):

        num_normals = np.random.randint(2, 11)  
        total_positions = (self.width + 1) * (self.height + 1)
        num_normals = min(num_normals, total_positions)
        selected_indices = np.random.choice(total_positions, size=num_normals, replace=False)
        normals = np.zeros((self.width + 1, self.height + 1, 2), dtype=np.float32)
        
        for idx in selected_indices:
            x, y = divmod(idx, self.height + 1)
            axis = np.random.choice(['x', 'y'])
            direction = np.random.choice([1, -1])
            if axis == 'x':
                normals[x, y, 0] = direction * 1.0
            else:
                normals[x, y, 1] = direction * 1.0
            forces_np = self.forces.cpu().numpy()
        for x in range(self.width + 1):
            for y in range(self.height + 1):
                if np.any(forces_np[x, y] != 0):
                    normals[x, y] = 0
        return torch.tensor(normals, dtype=torch.float32, device=self.device)


    def calculate_total_density(self):
        return np.mean(self.state.cpu().numpy())

    def is_connected(self, state, threshold=0.8):
        grid = state.cpu().numpy().reshape(self.height, self.width)
        G = nx.Graph()
        for i in range(self.height):
            for j in range(self.width):
                if grid[i, j] > threshold:
                    node = i * self.width + j
                    G.add_node(node)
                    # Add edges to neighboring occupied cells (4-connectivity)
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.height and 0 <= nj < self.width and grid[ni, nj] > threshold:
                            neighbor = ni * self.width + nj
                            G.add_edge(node, neighbor)
        # Check if all occupied cells form a single connected component
        return nx.is_connected(G) if len(G) > 0 else False

    def find_isolated_cells(self, state, width, height, high_threshold=0.8, low_threshold=0.2):
        """
        Find cells that are isolated, i.e., have high density but no neighbors with sufficient density.
        """
        isolated_cells = []
        state_np = state.reshape((width, height))
        for i in range(width):
            for j in range(height):
                if state_np[i, j] > high_threshold:
                    # Check if any neighbor has density > low_threshold
                    has_neighbor = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < width and 0 <= nj < height and state_np[ni, nj] > low_threshold:
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break
                    if not has_neighbor:
                        isolated_cells.append(i * width + j)
        return isolated_cells


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

        # Current density of the selected cell
        current_density = self.state[cell].item()

        # Initialize reward
        reward = 0.0
        info = {}

        if direction == 'increase':
            new_density = min(current_density + self.step_size, max_density)
            if new_density == current_density:
                reward -= 0.1
            else:
                self.state[cell] = new_density
                reward += 0.05  
        else:  
            new_density = max(current_density - self.step_size, min_density)
            if new_density == current_density:
                reward -= 0.1
            else:
                self.state[cell] = new_density
                reward += 0.05  

        self.current_step += 1

        # Calculate additional reward components
        additional_reward, reward_info = self.calculate_reward()
        reward += additional_reward
        reward = reward / (abs(reward) + 1)

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

        return self.state.cpu().numpy(), reward, done, False, info
  

    def calculate_reward(self):

        with torch.no_grad():
            compliance, constraint = fem.optim(args=self.args, x=self.state.cpu().numpy())
        #if compliance > 500:
        #    return -1
        current_density = self.calculate_total_density()

        w_compliance = self.w_compliance
        w_density_high = self.w_density_high  
        w_density_low = self.w_density_low
        w_total_mass = self.w_total_mass
        w_entropy = self.w_entropy           

        reward_compliance = -w_compliance * (compliance / 100.0)


        densities = self.state.cpu().numpy()

        density_high_reward = np.mean( (densities - 0.5) ** 2)  
        reward_density_high = w_density_high * density_high_reward

        density_low_reward = np.mean((0.5 - densities) ** 2)  
        reward_density_low = w_density_low * density_low_reward

        target_total_density = self.optimal_density * self.width * self.height
        total_mass = np.sum(densities)
        mass_deviation = max(total_mass - target_total_density, 0)
        reward_total_mass = -w_total_mass * mass_deviation


        density_entropy = -np.mean(densities * np.log(densities + 1e-8) + 
                                (1 - densities) * np.log(1 - densities + 1e-8))
        reward_entropy = w_entropy * density_entropy

        reward = (reward_compliance + #reward_connectivity + reward_isolated +
                reward_density_high + reward_density_low + reward_total_mass +
                reward_entropy)
        

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
            grid = self.state.cpu().numpy().reshape(self.height, self.width)
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
