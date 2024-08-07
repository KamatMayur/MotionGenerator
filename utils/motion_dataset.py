import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from utils.dual_quat import mat2dq

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
    sampling_window (int): Size of the sub-sampled window. Default is 12.
    center (int): index of the center frame (0 indexed)
    """
    def __init__(self, root_velocities_path, joint_velocities_path, 
                 root_transforms_path, joint_transforms_path, phase_manifolds_path, 
                 sub_sampling: int = 5, sampling_window: int = 12, center: int = 6):
        
        self.root_velocities = np.load(root_velocities_path).astype(np.float32)
        self.joint_velocities = np.load(joint_velocities_path).astype(np.float32)
        self.root_transforms = np.load(root_transforms_path).astype(np.float32)
        self.joint_transforms = np.load(joint_transforms_path).astype(np.float32)
        self.phase_manifolds = np.load(phase_manifolds_path).astype(np.float32)

        self.sub_sampling = sub_sampling
        self.sampling_window = sampling_window
        self.center = center
        self.num_frames = len(self.root_velocities)

    def __len__(self):
        return self.num_frames- ((self.sampling_window * self.sub_sampling) - self.sub_sampling)
    
    def get_sampling_window_inds(self, n):
        if self.num_frames >= self.sampling_window * self.sub_sampling:
            return [(i * self.sub_sampling) + n for i in range(self.sampling_window)]

    def __getitem__(self, idx):

        inds = self.get_sampling_window_inds(idx)

        # Root transforms relative to the center frame
        rt = self.root_transforms[inds]
        cf_rt_inv = np.linalg.inv(rt[self.center])
        cf_rel_rt = cf_rt_inv @ rt

        jt = self.joint_transforms[inds]
        
        # convert to dual quaternions
        cf_rel_rdq = mat2dq(torch.tensor(cf_rel_rt))
        jdq = mat2dq(torch.tensor(jt))



        

        return (self.phase_manifolds[inds],
                self.root_velocities[inds],
                self.joint_velocities[inds],
                cf_rel_rdq,
                jdq
                )
    