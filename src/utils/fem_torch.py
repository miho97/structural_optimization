import torch
import numpy as np
import time

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()


def mbb_beam_1(width=6, height=6, density=0.5, y=1, x=0):  

    normals = torch.zeros((width + 1, height + 1, 2), dtype=torch.float32, device= device)
    normals[-1, -1, y] = 1
    normals[0, :, x] = 1
    forces = torch.zeros((width + 1, height + 1, 2), dtype=torch.float32, device= device)
    forces[0, 0, y] = -1
    return normals, forces, density

def mbb_beam_2(width=6, height=6, density=0.4, y=1, x=0):  
    normals = torch.zeros((width + 1, height + 1, 2), dtype=torch.float32, device= device)
    normals[-1, -1, y] = 1
    normals[0, :, x] = 1
    forces = torch.zeros((width + 1, height + 1, 2), dtype=torch.float32, device= device)
    forces[0, height, y] = -1
    return normals, forces, density


def mbb_beam_3(width=6, height=6, density=0.4, y=1, x=0):  
    normals = torch.zeros((width + 1, height + 1, 2), dtype=torch.float32, device= device)
    normals[-1, -1, y] = 1
    normals[0, :, x] = 1
    forces = torch.zeros((width + 1, height + 1, 2), dtype=torch.float32, device= device)
    forces[width//2, 0, y] = -1
    return normals, forces, density

def mbb_beam_4(width=6, height=6, density=0.4, y=1, x=0):  
    normals = torch.zeros((width + 1, height + 1, 2), dtype=torch.float32, device= device)
    normals[-1, -1, y] = 1
    normals[0, :, x] = 1
    forces = torch.zeros((width + 1, height + 1, 2), dtype=torch.float32, device= device)
    forces[int(width//1.5), 0, y] = -1
    return normals, forces, density


def mbb_beam_5(width=6, height=6, density=0.4, y=1, x=0):  
    normals = torch.zeros((width + 1, height + 1, 2), dtype=torch.float32, device= device)
    normals[0, height, y] = 1
    normals[width, :, x] = 1
    forces = torch.zeros((width + 1, height + 1, 2), dtype=torch.float32, device= device)
    forces[width, 0, y] = -1
    return normals, forces, density

def get_args(normals, forces, density=0.4):
    """
    Extract parameters from normals and forces.
    """
    width = normals.shape[0] - 1
    height = normals.shape[1] - 1
    fixdofs = torch.nonzero(normals.view(-1) > 0, as_tuple=False).squeeze(-1)
    alldofs = torch.arange(2 * (width + 1) * (height + 1), device= device)
    freedofs = torch.tensor(list(set(alldofs.cpu().numpy()) - set(fixdofs.cpu().numpy())), dtype=torch.long, device= device)
    params = {
        # Material properties
        'young': 1.0, 
        'young_min': 1e-9, 
        'poisson': 0.3, 
        'g': 0.0,
        # Constraints
        'density': density, 
        'xmin': 0.001, 
        'xmax': 1.0,
        # Input parameters
        'nelx': width, 
        'nely': height, 
        'mask': 1.0, 
        'penal': 3.0, 
        'filter_width': 1.0,
        'freedofs': freedofs, 
        'fixdofs': fixdofs, 
        'forces': forces.view(-1),
        # Optimization parameters
        'opt_steps': 80, 
        'print_every': 20
    }
    return params

def young_modulus(x, e_0=1.0, e_min=1e-9, p=3.0):
    """
    Compute Young's modulus based on density.
    """
    return e_min + x**p * (e_0 - e_min)

def physical_density(x, args, volume_constraint=False, use_filter=False):
    """
    Apply physical density filtering.
    """
    x = args['mask'] * x.view(args['nely'], args['nelx'])  # Reshape to 2D
    if use_filter:
        return gaussian_filter(x, args['filter_width'])
    else:
        return x

def gaussian_filter(x, width):
    """
    Apply a simple average filter (can be replaced with a Gaussian filter).
    """
    kernel_size = int(width * 2 + 1)
    padding = width
    return torch.nn.functional.avg_pool2d(x.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, padding=padding).squeeze()

def get_stiffness_matrix(e, nu):
    """
    Generates a simplified stiffness matrix for each element.
    """
    k = torch.tensor([
        1/2 - nu/6, 
        1/8 + nu/8, 
        -1/4 - nu/12, 
        -1/8 + 3*nu/8,
        -1/4 + nu/12, 
        -1/8 - nu/8, 
        nu/6, 
        1/8 - 3*nu/8
    ], dtype=torch.float32).to(device=device)
    k = e / (1 - nu**2) * torch.tensor([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ], dtype=torch.float32, device = device)
    return k

def get_k(stiffness, ke, nely, nelx):
    """
    Assemble the global stiffness matrix K based on element connectivity.
    """
    # Generate meshgrid for elements
    ely = torch.arange(nely, device= device)
    elx = torch.arange(nelx, device= device)
    ely, elx = torch.meshgrid(ely, elx, indexing='ij')
    ely = ely.flatten()
    elx = elx.flatten()
    
    n_elements = elx.shape[0]
    
    # Node indices
    n1 = (nely + 1) * elx + ely
    n2 = (nely + 1) * (elx + 1) + ely
    n3 = (nely + 1) * (elx + 1) + (ely + 1)
    n4 = (nely + 1) * elx + (ely + 1)
    
    # DOFs for each element
    edof = torch.stack([
        2 * n1, 2 * n1 + 1,
        2 * n2, 2 * n2 + 1,
        2 * n3, 2 * n3 + 1,
        2 * n4, 2 * n4 + 1
    ], dim=1)  # Shape: (num_elements, 8)
    
    # Flattened DOFs for vectorized operations
    edof_i = edof.unsqueeze(2).repeat(1, 1, 8).flatten()
    edof_j = edof.unsqueeze(1).repeat(1, 8, 1).flatten()
    
    # Expand stiffness and ke for element-wise multiplication
    stiffness = stiffness.view(-1, 1, 1)  # Shape: (num_elements, 1, 1)
    ke = ke.view(1, 8, 8)                # Shape: (1, 8, 8)
    
    # Compute values to add to K
    values = (stiffness * ke).flatten()   # Shape: (num_elements * 64,)
    
    # Initialize K
    size = 2 * (nelx + 1) * (nely + 1)
    K = torch.zeros((size, size), dtype=torch.float32, device= device)
    
    # Assemble K using scatter_add
    K.index_put_((edof_i, edof_j), values, accumulate=True)
    
    return K

def displace(x_phys, ke, forces, freedofs, fixdofs, penal=3.0, e_min=1e-9, e_0=1.0, nely=6, nelx=6):
    """
    Computes displacements using FEM.
    """
    stiffness = young_modulus(x_phys, e_0, e_min, penal)
    K = get_k(stiffness, ke, nely, nelx)
    
    # Apply boundary conditions by modifying K and F
    K = K.clone()
    F = forces.clone()
    
    # Fix DOFs
    for dof in fixdofs:
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1
        F[dof] = 0
    
    # Solve for displacements
    try:
        U = torch.linalg.solve(K, F)
    except RuntimeError:
        # In case K is singular, return a high compliance
        return torch.full((K.shape[0],), 1e6, dtype=torch.float32, device=device)
    
    return U

def compliance_calc(u, forces):
    """
    Calculate compliance as F^T * U.
    """
    return torch.dot(forces, u)

def objective_calc(x, args, volume_constraint=False, use_filter=False):
    """
    Objective function to calculate compliance.
    """
    x_phys = physical_density(x, args, volume_constraint, use_filter)
    ke = get_stiffness_matrix(args['young'], args['poisson'])
    u = displace(x_phys, ke, args['forces'], args['freedofs'], args['fixdofs'], 
                penal=args['penal'], e_min=args['young_min'], e_0=args['young'], 
                nely=args['nely'], nelx=args['nelx'])
    c = compliance_calc(u, args['forces'])
    return c

def compliance_and_constraint(x, args, volume_constraint=False, use_filter=False):
 
    compliance = objective_calc(x, args, volume_constraint, use_filter)
    total_mass = torch.mean(x)
    return compliance, total_mass

def _get_dof_indices(freedofs, fixdofs, k_xlist, k_ylist):

    concatenated = torch.cat([freedofs, fixdofs], dim=0)
    
    index_map = inverse_permutation(concatenated)
    
    # Create a boolean mask where both k_xlist and k_ylist are in freed DOFs
    keep = torch.isin(k_xlist, freedofs) & torch.isin(k_ylist, freedofs)
    
    # Apply the mask to index_map for k_ylist and k_xlist
    i = index_map[k_ylist][keep]
    j = index_map[k_xlist][keep]
    
    # Stack the filtered indices
    ij = torch.stack([i, j], dim=0)
    
    return index_map, keep, ij

import torch

def mean_density(x, args, volume_constraint=False, use_filter=False):

    if x.dim() > 1:
        x = x.view(-1)
    
    # Compute physical density
    phys_density = physical_density(x, args, volume_constraint, use_filter)
    
    # Compute mean density
    mean_phys_density = torch.mean(phys_density)
    
    # Compute mean of the mask
    mean_mask = torch.mean(args['mask'])
    
    # Normalize mean density by mean mask
    normalized_mean_density = mean_phys_density / mean_mask
    
    return normalized_mean_density


def fast_stopt_pytorch(args, x=None, verbose=True, learning_rate=1e-3, num_epochs=1000):

    if x is None:
        x = torch.ones((args['nely'], args['nelx']), dtype=torch.float32, device='cpu') * args['density']
        x = x.view(-1)
    else:
        x = x.clone().detach().to('cpu')

    # Set requires_grad to True for optimization
    x = x.clone().detach().requires_grad_(True)

    # Define optimizer
    optimizer = torch.optim.Adam([x], lr=learning_rate)

    # Define loss function components
    def reshape_density(x):
        return x.view(args['nely'], args['nelx'])

    def objective_fn(x):
        return objective_calc(reshape_density(x), args)

    def constraint_fn(x):
        return mean_density(x, args) - args['density']

    # Initialize lists to store losses and frames
    losses = []
    frames = []
    mass_constraints = []

    # Start optimization loop
    start_time = time.time()
    print('Optimizing a problem with {} nodes'.format(len(args['forces'])))

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        # Compute compliance
        compliance = objective_fn(x)

        # Compute mass and constraint
        mass = mean_density(x, args)
        mass_constraint = torch.relu(mass - args['density'])

        # Define loss with penalty for mass constraint
        loss = compliance + 1000.0 * mass_constraint  # lambda_mass = 1000.0

        # Backward pass
        loss.backward()
        optimizer.step()

        # Detach and store results
        loss_val = loss.item()
        compliance_val = compliance.item()
        mass_constraint_val = mass_constraint.item()

        losses.append(loss_val)
        frames.append(x.detach().cpu().numpy().reshape(args['nely'], args['nelx']))
        mass_constraints.append(mass_constraint_val)

        # Logging
        if verbose and (epoch == 1 or epoch % args['print_every'] == 0):
            elapsed_time = time.time() - start_time
            print(f'Epoch {epoch}, Loss: {loss_val:.2e}, Compliance: {compliance_val:.2e}, Mass Constraint: {mass_constraint_val:.2e}, Time Elapsed: {elapsed_time:.2f}s')

    final_constraint = mass_constraints[-1]
    final_x = x.detach().cpu().numpy().reshape(args['nely'], args['nelx'])
    frames = np.array(frames)
    losses = np.array(losses)

    return losses, final_x, frames, final_constraint

def inverse_permutation(indices):

    inverse_perm = torch.zeros(len(indices), dtype=torch.long, device=indices.device)
    inverse_perm[indices] = torch.arange(len(indices), dtype=torch.long, device=indices.device)
    return inverse_perm


def main():
    # Example usage
    width, height, density = 6, 6, 0.4
    normals, forces, density = mbb_beam_1(width, height, density)
    args = get_args(normals, forces, density)
    args['nely'] = height
    args['nelx'] = width
    
    # Initialize density distribution
    x = torch.ones((args['nely'], args['nelx']), dtype=torch.float32, device='cpu') * args['density']
    x = torch.tensor([
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.54856173, 1.0, 1.0, 0.63878568],
        [0.0, 0.0, 0.0, 0.81265539, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ])

    x = x.view(-1)
    
    # Compute compliance and mass
    compliance, mass = compliance_and_constraint(x, args)
    print(f"Compliance: {compliance.item():.4f}, Mass: {mass.item():.4f}")

if __name__ == "__main__":
    main()