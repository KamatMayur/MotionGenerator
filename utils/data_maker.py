import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from pathlib import Path
from scipy.spatial.transform import Rotation as R


FPS = 30.0
MAIN_BONES = [
    "root",
    "pelvis",
    "spine_01",
    "spine_02",
    "spine_03",
    "spine_04",
    "spine_05",
    "clavicle_l",
    "upperarm_l",
    "lowerarm_l",
    "hand_l",
    "clavicle_r",
    "upperarm_r",
    "lowerarm_r",
    "hand_r",
    "neck_01",
    "neck_02",
    "head",
    "thigh_l",
	"calf_l",
    "foot_l",
    "ball_l",
    "thigh_r",
    "calf_r",
    "foot_r",
    "ball_r"
]

def plot_frame(global_poses, main_bone_indices, parent_indices, frame_index=0):
    # Extract positions from global transformations
    positions = global_poses[frame_index, :, :3, 3]

    # Plot the skeleton
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in main_bone_indices:
        if parent_indices[i] != -1:
            parent_pos = positions[parent_indices[i]]
            ax.plot([positions[i][0], parent_pos[0]], [positions[i][1], parent_pos[1]], [positions[i][2], parent_pos[2]], 'bo-', markersize=3)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    center_x = (np.max(positions[:, 0]) + np.min(positions[:, 0])) / 2
    center_y = (np.max(positions[:, 1]) + np.min(positions[:, 1])) / 2

    ax.set_xlim([center_x-1, center_x+1])
    ax.set_ylim([center_y-1, center_y+1])
    ax.set_zlim([0,2])
    ax.set_aspect("equal")
    plt.show()

def plot_frame_sequence(global_poses, main_bone_indices, parent_indices,
                        line_style='-', line_color="g",
                        marker_style='o', marker_color="y", marker_size=3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_frames = global_poses.shape[0]
    root_index = main_bone_indices[0]  # Assuming the root is the first in main_bone_indices

    for frame_index in range(num_frames):
        # Extract positions from global transformations
        positions = global_poses[frame_index, :, :3, 3]
        rotations = global_poses[frame_index, :, :3, :3]
        alpha = frame_index / num_frames

        for i in main_bone_indices:
            if parent_indices[i] != -1:
                parent_pos = positions[parent_indices[i]]
                ax.plot([positions[i][0], parent_pos[0]], [positions[i][1], parent_pos[1]], [positions[i][2], parent_pos[2]],
                        linestyle=line_style, color=line_color, alpha=alpha,
                        marker=marker_style, markersize=marker_size, markerfacecolor=marker_color)

        # Plot arrow for the root position
        root_pos = positions[root_index]
        root_x_dir = rotations[root_index, :, 1]  # Extract y direction of root rotation
        ax.quiver(root_pos[0], root_pos[1], root_pos[2],
                  root_x_dir[0], root_x_dir[1], root_x_dir[2],
                  color='r', length=0.1, normalize=True, arrow_length_ratio=0.1, linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Calculate the center and set limits
    all_positions = global_poses[:, :, :3, 3].reshape(-1, 3)
    center_x = (np.max(all_positions[:, 0]) + np.min(all_positions[:, 0])) / 2
    center_y = (np.max(all_positions[:, 1]) + np.min(all_positions[:, 1])) / 2
    center_z = (np.max(all_positions[:, 2]) + np.min(all_positions[:, 2])) / 2
    max_range = max(np.ptp(all_positions[:, 0]), np.ptp(all_positions[:, 1]), np.ptp(all_positions[:, 2]))

    ax.set_xlim([center_x - max_range / 2, center_x + max_range / 2])
    ax.set_ylim([center_y - max_range / 2, center_y + max_range / 2])
    ax.set_zlim([center_z - max_range / 2, center_z + max_range / 2])
    ax.set_aspect('auto')

    plt.show()

def load_animation(file_path, rest_pose, parent_indices):
    # Load animation JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)
    poses = np.array(data['Frames']).transpose(0, 1, 3, 2)[30:]


    # Positions are scaled down from centimeters to meters
    poses[:, :, :3, 3] *= 0.01


    # Array to store the global transfroms of all joints for all frames
    global_poses = np.zeros_like(poses)
    # Array to store the rest pose relative local transfroms of all joints for all frames
    # These are local poses relative to their parent and are offsets wrt to rest pose.
    rest_rel_local_poses = np.matmul(np.expand_dims(np.linalg.inv(rest_pose), axis=0), poses)
    # Array to store root relative transforms for joints for all frames (all transforms are wrt to the root transform)
    # The root transform will become identity
    root_rel_poses = np.zeros_like(global_poses)

    num_frames, num_bones, _, _ = poses.shape
    local_poses = np.tile(rest_pose, (num_frames, 1, 1, 1)) @ rest_rel_local_poses

    # Initialize the global transform for the first bone (assuming it has no parent)
    global_poses[:, 0] = local_poses[:, 0]  # Assuming the first bone has no parent (parent_indices[0] == -1)
    
    # Use a loop to compute global transforms for the remaining bones for all frames
    for bone_idx in range(1, num_bones):
        # Get the parent index for the current bone
        parent_idx = parent_indices[bone_idx]
        
        # Compute the global transform for the current bone by multiplying the parent's global transform with the current local transform
        global_poses[:, bone_idx] = global_poses[:, parent_idx] @ local_poses[:, bone_idx]
    
    # Root rel poses are calculated by post multiplying the global poses of all joints with the inverse of root joint
    root_rel_poses = np.matmul(np.linalg.inv(global_poses[:, [0]]), global_poses)

    return rest_rel_local_poses, global_poses, root_rel_poses
    
def load_skeleton(file_path):
    # Load skeleton JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # bones = data['bones']
    bone_names = data["BoneNames"]
    parent_indices = data['ParentIndices']
    bone_index_map = {bone: bone_names.index(bone) for bone in MAIN_BONES if bone in bone_names}
    main_bone_indices = np.array(list(bone_index_map.values()))
    bone_transforms = np.array(data['BoneTransforms']).transpose(0,2,1)

    # Positions are scaled down from centimeters to meters
    bone_transforms[:, :3, 3] *= 0.01
 

    return bone_transforms, parent_indices, main_bone_indices, bone_index_map

def get_root_velocities(global_poses, fps):
    num_frames, _, _, _ = global_poses.shape
    root_positions = global_poses[:, 0, :3, 3]
    root_velocities = np.zeros_like(root_positions)
    root_velocities[1:] = np.diff(root_positions, axis=0) * fps
    return root_velocities

def get_joint_velocitites(root_rel_poses, main_bone_indices, fps):

    joint_positions = root_rel_poses[:, :, :3, 3][:, main_bone_indices]
    joint_velocities = np.zeros_like(joint_positions)
    joint_velocities[1:] = np.diff(joint_positions, axis=0) * fps
    return joint_velocities[:, 1:]

def main():
    animation_folder_path = Path("animations")
    skeleton_path = "skeleton/SK_UEFN_Mannequin.json"
    data_path = "data/"

    animation_files = list(animation_folder_path.glob('*.json'))

    rest_pose, parent_indices, main_bone_indices, bone_index_map = load_skeleton(skeleton_path)


    root_velocities = np.zeros((0, 3))
    joint_velocities = np.zeros((0, len(main_bone_indices) - 1, 3))
    root_transforms = np.zeros((0, 4,4))
    joint_transforms = np.zeros((0, len(main_bone_indices)-1, 4, 4))

    for animation in animation_files[4:]:
        rest_rel_local_poses, global_poses, root_rel_poses = load_animation(animation, rest_pose, parent_indices)

        root_vels = get_root_velocities(global_poses, 30)
        joint_vels = get_joint_velocitites(root_rel_poses, main_bone_indices, 30)
        root_trans = global_poses[:, 0]
        joint_trans = root_rel_poses[:, main_bone_indices[1:]]

        root_velocities = np.append(root_velocities, root_vels, axis=0)
        joint_velocities = np.append(joint_velocities, joint_vels, axis=0)
        root_transforms = np.append(root_transforms, root_trans, axis=0)
        joint_transforms = np.append(joint_transforms, joint_trans, axis=0)

    parent_indices = np.array([
                    -1 if parent_indices[child] == -1 else
                    list(main_bone_indices).index(parent_indices[child])
                    for child in main_bone_indices
                    ])
    

    # UNCOMMENT TO SAVE: BE CAREFUL AS IT MIGHT OVERWRITE THE EXISTING DATA
    np.save(data_path + "root_velocities.npy", root_velocities.astype(np.float32))
    np.save(data_path + "joint_velocities.npy", joint_velocities.astype(np.float32))
    np.save(data_path + "root_transforms.npy", root_transforms.astype(np.float32))
    np.save(data_path + "joint_transforms.npy", joint_transforms.astype(np.float32))
    np.save(data_path + "parent_indices.npy", parent_indices.astype(np.int8))


if __name__ == '__main__':
    main()
        









