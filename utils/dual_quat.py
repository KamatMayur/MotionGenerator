import torch
import torch.nn.functional as F

"""
PyTorch Autograd compatible maths for quaternions and dual quaternions
"""

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def mat2q(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """

    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def q2mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    ab =  torch.stack((ow, ox, oy, oz), -1)
    return ab


def qinv(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


def qconj(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Return the conjugate of the quaternions.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Conjugate of the quaternion (..., 4).
    """
    conj = quaternions.clone()
    conj[..., 1:] = -conj[..., 1:]
   
    return conj


def mat2dq(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert homogeneous transforms to dual quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 4, 4).

    Returns:
        dual quaternions with non-dual part (rotation) first, as tensor of shape (..., 8).
    """

    q_r = mat2q(matrix[..., :3, :3])

    t = matrix[..., :3, 3]
    zeros = torch.zeros(t.shape[:-1] + (1,))

    q_t = torch.cat((zeros, t), dim=-1)

    q_d = 0.5 * qmul(q_t, q_r)

    dq = torch.cat((q_r, q_d), dim=-1)

    return dq


def dq2mat(dual_quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert dual quaternions given homogenous transformation matrices.

    Args:
        dual_quaternions: quaternions with non-dual part (rotation) first first,
            as tensor of shape (..., 8).

    Returns:
        Rotation matrices as tensor of shape (..., 4, 4).
    """
    batch_dims = dual_quaternions.shape[:-1]
    identity = torch.eye(4)
    mats = torch.tile(identity, (batch_dims + (1, 1)))
    
    mats[..., :3, :3] = q2mat(dual_quaternions[..., :4])
    t_h = qmul((2 * dual_quaternions[..., 4:]), qconj(dual_quaternions[..., :4]))
    mats[..., :3, 3] = t_h[...,1:]
    
    return mats


def dqmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two dual quaternions representing transformations, returning the dual quaternion
    representing their composition.

    Args:
        a: Dual quaternions as tensor of shape (..., 8), real part first.
        b: Dual quaternions as tensor of shape (..., 8), real part first.

    Returns:
        The product of a and b, a tensor of dual quaternions of shape (..., 8).
    """
    q_r = qmul(a[..., :4], b[..., :4])
    q_d = qmul(a[..., :4], b[..., 4:]) + qmul(a[..., 4:], b[..., :4])

    dq = torch.cat((q_r, q_d), dim=-1)

    return dq


def dqinv(dq: torch.Tensor) -> torch.Tensor:
    """
    Calculate the inverse of a dual quaternion.

    Args:
        dq: Dual quaternions as tensor of shape (..., 8), with real part first,
            representing a dual quaternion.

    Returns:
        The inverse of the input dual quaternion, a tensor of dual quaternions
        of shape (..., 8).
    """
    
    q_r_inv = qinv(dq[..., :4])
    q_d_inv = qmul(qmul(-q_r_inv, dq[..., 4:]), q_r_inv)

    dq_inv = torch.cat((q_r_inv, q_d_inv), dim=-1)

    return dq_inv

    
def dqnorm(dq):
        """
        This function enforces the unitary conditions for valid dual quaternions:
        1. The real part should have a norm of 1.
        2. The real and the dual parts should be orthogonal.

        Parameters:
            dq (torch.Tensor): A tensor containing dual quaternions of shape(..., 8). 
                            The first 4 components are for the real quaternion, and the remaining 4 are for the 
                            dual quaternion.

        Returns:
            torch.Tensor: The normalized quaternion.
        """

        real = dq[..., :4]  # (batch_size, num_frames, num_bones, 4)
        dual = dq[..., 4:]  # (batch_size, num_frames, num_bones, 4)

        # Normalize the real part
        real_norm = torch.norm(real, dim=-1, keepdim=True)
        real_normalized = real / real_norm

        # Project the dual part onto the real part
        real_dot_dual = torch.sum(real_normalized * dual, dim=-1, keepdim=True)  # (batch_size, num_frames, num_bones, 1)
        
        # Orthogonalize the dual part if necessary
        dual_orthogonal = dual - real_dot_dual * real_normalized

        dual_orthonormal = F.normalize(dual_orthogonal, dim=-1)

        # Scaling the dual part to have the original magnitude
        dual_orthogonal = dual_orthonormal * torch.norm(dual, dim=-1, keepdim=True)
    
        # Combine the normalized real and dual parts
        normalized_dq = torch.cat([real_normalized, dual_orthogonal], dim=-1)

        return normalized_dq


def dqeye(*shape):
    """
    Generate an identity dual quaternion tensor with the given shape, 
    where the last dimension is fixed at 8.

    Parameters:
    shape (tuple): The first n-1 dimensions of the desired output tensor.

    Returns:
    torch.Tensor: A tensor of identity dual quaternions with shape (*shape, 8).
    """
    # Create a tensor of zeros with the desired shape and last dimension 8
    identity_dq = torch.zeros(*shape, 8)
    
    # Set the first element of the last dimension (real part of the quaternion) to 1
    identity_dq[..., 0] = 1
    
    return identity_dq

# t =  torch.rand(32, 3, 5, 8)


# print(dqnorm(t).shape)
