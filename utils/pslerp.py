import torch

def slerp(p0, p1, t):
    """ 
    Perform Spherical Linear Interpolation between two vectors p0 and p1.
    
    Parameters:
    p0 (torch.Tensor): Start vector of shape (batch_size, n, 2)
    p1 (torch.Tensor): End vector of shape (batch_size, n, 2)
    t (float or torch.Tensor): Interpolation factor (0 <= t <= 1)
    
    Returns:
    torch.Tensor: Interpolated vectors of shape (batch_size, n, 2)
    """
    dot = (p0 * p1).sum(dim=-1, keepdim=True).clamp(-0.9999, 0.9999)  # Dot product for each vector pair

    assert not torch.isnan(dot).any(), "NaNs detected in dot product"
    theta_0 = torch.acos(dot)
    assert not torch.isnan(theta_0).any(), "NaNs detected in acos(dots) product"

    sin_theta_0 = torch.sin(theta_0)

    # Handle small angles separately to avoid division by zero
    small_angle = sin_theta_0 < 1e-6
    result = torch.where(
        small_angle,
        (1.0 - t) * p0 + t * p1,
        (torch.sin(theta_0 - theta_0 * t) * p0 + torch.sin(theta_0 * t) * p1) / sin_theta_0
    )

    return result


def pslerp(a1, f1, p0, p1, delta_t, weight):
    """
    Apply pslerp transformation to the vectors p0 and p1.
    
    Parameters:
    a1 (torch.Tensor): Amplitudes at time t+delta_t for n channels of shape (batch_size, n, 1)
    f1 (float or torch.Tensor): Frequencies at time t+delta_t for n channels of shape (batch_size, n, 1)
    p0 (torch.Tensor): phases at time t of shape (batch_size, n, 2)
    p1 (torch.Tensor): phases at time t+delta_t of shape (batch_size, n, 2)
    delta_t (float or torch.Tensor): Time delta
    weight (float or torch.Tensor): Weight for slerp interpolation
    
    Returns:
    torch.Tensor: Interpolated phases of shape (n, 2)
    """

    # Ensure delta_t is a tensor if passed as a float
    if isinstance(delta_t, float):
        delta_t = torch.tensor(delta_t, device=p0.device)
    if isinstance(weight, float):
        weight = torch.tensor(weight, device=p0.device)

    def rot_mat(t):
        cos_t = torch.cos(t)
        sin_t = torch.sin(t)

        return torch.stack([
                torch.stack([cos_t, -sin_t], dim=-1),
                torch.stack([sin_t, cos_t], dim=-1)
            ], dim=-2)

    # Calculate the rotation matrix for each vector
    thetas = delta_t * 2 * torch.pi * f1  # Shape (n, 1)
  
    rot_mats = torch.stack([rot_mat(t) for t in thetas], dim=-3)
    rot_mats = rot_mats.view(thetas.shape[0], thetas.shape[1], 2, 2)  # Reshape to (b, n, 2, 2)

    # p0 = torch.stack([rot_mats[i] @ p0[i] for i in range(p0.shape[0])])
    p0 = torch.einsum('bnij,bnj->bni', rot_mats, p0)
   
    pslerped = a1.unsqueeze(-1) * slerp(p0, p1, weight)

    return pslerped

# Test
import numpy as np

pm0 = torch.tensor(np.random.uniform(-1, 1, 10024*2*5), requires_grad=True)
pm1 = torch.tensor(np.random.uniform(-1, 1, 10024*2*5), requires_grad=True)
pm0 = pm0.reshape(10024, 5, 2)
pm1 = pm1.reshape(10024, 5, 2)

a1 = torch.tensor(np.random.uniform(0, 2, (10024, 5)), requires_grad=True)
f1 = torch.tensor(np.random.uniform(1, 3, (10024, 5)), requires_grad=True)

delta_t = torch.tensor(1.0/6, requires_grad=True)

weight = torch.tensor(0.5, requires_grad=True)

pmout = pslerp(a1, f1, pm0, pm1, delta_t, weight)

print(pm0.shape, pm1.shape, a1.shape, f1.shape)
print(pmout.shape)

pmout.sum().backward()
