import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from utils.dual_quat import mat2dq
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class MotionDataset(Dataset):
    """
    Parameters:
    root_velocities_path (str): Path to the root velocities data file.
    joint_velocities_path (str): Path to the joint velocities data file.
    root_transforms_path (str): Path to the root transforms data file.
    joint_transforms_path (str): Path to the joint transforms data file.
    phase_manifolds_path (str): Path to the phase manifolds data file.
    sub_sampling (int): Additional sampling done over the actual samples. 
                        1 means original sampling, n means every nth sample from the original samples. Default is 5.
    sampling_window (int): Size of the sub-sampled window for trajectories. Default is 13.
    sequence_length (int): Number of poses to be used in regression. Default is 6.
    center (int): index of the center frame (0 indexed)

    NOTE: This class assumes original sampling rate of 30.0 fps of the data which is used for calculations inside!
    """
    def __init__(self, root_velocities_path, joint_velocities_path, 
                 root_transforms_path, joint_transforms_path, phase_manifolds_path, 
                 sub_sampling: int = 5, sampling_window: int = 13, sequence_length: int = 6, center: int = 6):
        
        self.root_velocities = np.load(root_velocities_path).astype(np.float32)
        self.joint_velocities = np.load(joint_velocities_path).astype(np.float32)
        self.root_transforms = np.load(root_transforms_path).astype(np.float32)
        self.joint_transforms = np.load(joint_transforms_path).astype(np.float32)
        self.phase_manifolds = np.load(phase_manifolds_path).astype(np.float32)

        self.sub_sampling = sub_sampling
        self.sampling_window = sampling_window
        self.sequence_length = sequence_length
        self.center = center
        self.num_frames = len(self.root_velocities)

    def __len__(self):
        return (self.num_frames- ((self.sampling_window + self.sequence_length - 1) * self.sub_sampling)) + self.sub_sampling
    
    def get_sampling_window_inds(self, n):
        """
        Returns the indices for sampling a window of frames starting from
        index n in the originally sampled data.
        
        Parameters:
        - n: index of the frame in the original data.

        Returns:
        - List of indices for sampling.
        """
        if self.num_frames >= self.sampling_window * self.sub_sampling:
            return [n + (i * self.sub_sampling) for i in range(self.sampling_window + self.sequence_length-1)]
        
    def get_root_trajectories(self, indices):
        """
        Returns the input and output smoothed subsampled trajectories of the root,
        relative to the root transforms in the center frame.

        Parameters:
        - indices: List or array of indices to sample from the root transforms.
        
        Returns:
        - Tuple of Arrays of shape (sampling_window, 9) containing:
          1. Position (x, y, z)
          2. Facing direction (x, y) - The y direction of the root's rotation matrix
          3. Velocity (x, y, z)
          4. Speed - Magnitude of the velocity
        """

        vels = self.root_velocities[indices]
        speeds = np.linalg.norm(vels, axis=1, keepdims=True)

        cf_rel_rt = self.get_center_rel_root_transforms(indices)
        pos = cf_rel_rt[:, :3, 3]
        r = cf_rel_rt[:, :3, :3]

        dir = r[:, :3, 1] [:, :2]

        trajectories = np.hstack([pos, dir, vels, speeds])

        return trajectories

    def get_center_rel_root_transforms(self, indices):
        """
        Computes the root transforms relative to the center frame.
        
        Parameters:
        - indices: List or array of indices to sample the root transforms.

        Returns:
        - Array of shape (num_samples, 4, 4) with root transforms relative to the center frame.
        """

        rt = self.root_transforms[indices]
        cf_rt_inv = np.linalg.inv(rt[self.center])

        # Root transforms relative to the center frame
        cf_rel_rt = cf_rt_inv @ rt

        return cf_rel_rt

    def get_sequence(self, inds):
        rvs = np.zeros((self.sequence_length, ) + self.root_velocities[0].shape)
        jvs = np.zeros((self.sequence_length, ) + self.joint_velocities[0].shape)
        rts = np.zeros((self.sequence_length, ) + self.root_transforms[0].shape)
        jts = np.zeros((self.sequence_length, ) + self.joint_transforms[0].shape)
        trajs = np.zeros((self.sequence_length, self.sampling_window, 9))
        pms = np.zeros((self.sequence_length,) + self.phase_manifolds[0].shape)
        
        for i in range(len(inds) - self.sampling_window+1):
            ids = [inds[j] for j in range(i, i+self.sampling_window)]
            rvs[i] = self.root_velocities[ids][self.center]
            jvs[i] = self.joint_velocities[ids][self.center]
            rts[i] = self.root_transforms[ids][self.center]
            jts[i] = self.joint_transforms[ids][self.center]
            pms[i] = self.phase_manifolds[ids][self.center]
            trajs[i] = self.get_root_trajectories(ids)

        return rvs, jvs, rts, jts, pms, trajs

    def __getitem__(self, idx):

        inds = self.get_sampling_window_inds(idx)

        rvs, jvs, rts, jts, pms, trajs = self.get_sequence(inds)
        
        # convert to dual quaternions
        rdqs = mat2dq(torch.tensor(rts))
        jdqs = mat2dq(torch.tensor(jts))
        
        return (torch.tensor(rvs, dtype=torch.float32),
                torch.tensor(jvs, dtype=torch.float32),
                rdqs.to(dtype=torch.float32),
                jdqs.to(dtype=torch.float32),
                torch.tensor(pms, dtype=torch.float32),
                torch.tensor(trajs, dtype=torch.float32))
      
    
# dataset = MotionDataset(r"c:\Users\mayur\Documents\GitHub\MotionGenerator\data\root_velocities.npy",
#                         r"c:\Users\mayur\Documents\GitHub\MotionGenerator\data\joint_velocities.npy",
#                         r"c:\Users\mayur\Documents\GitHub\MotionGenerator\data\root_transforms.npy",
#                         r"c:\Users\mayur\Documents\GitHub\MotionGenerator\data\joint_transforms.npy",
#                         r"c:\Users\mayur\Documents\GitHub\MotionGenerator\data\phase_manifolds.npy",
#                         5, 13, 6, 6)


# dl = DataLoader(dataset, 128)

# dataset.__getitem__(dataset.__len__()-1)

# for d in dl:
#     print(d[0].shape)

