import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np

def plot_frame_sequence(ax, positions, parent_indices, center,
                        line_style='-', line_color=("r", "g"),
                        marker_style='o', marker_color=("r", "g"), marker_size=3):
   
    num_frames = positions.shape[0]
    num_bones = positions.shape[1]

    
    for frame_index in range(0, center+1):
        # Extract positions from global transformations
        pos = positions[frame_index]

        alpha = ((frame_index + 1 )/ num_frames) * 0.5

        for i in range(num_bones):
            if parent_indices[i] != -1:
                parent_pos = pos[parent_indices[i]]
        
                ax.plot([pos[i][0], parent_pos[0]], [pos[i][1], parent_pos[1]], [pos[i][2], parent_pos[2]],
                        linestyle=line_style, color=line_color[0], alpha=alpha,
                        marker=marker_style, markersize=marker_size, markerfacecolor=marker_color[0])
                
    for frame_index in range(center+1, num_frames):
        # Extract positions from global transformations
        pos = positions[frame_index]

        alpha = ((frame_index - center + 1 )/ num_frames) * 0.5

        for i in range(num_bones):
            if parent_indices[i] != -1:
                parent_pos = pos[parent_indices[i]]
        
                ax.plot([pos[i][0], parent_pos[0]], [pos[i][1], parent_pos[1]], [pos[i][2], parent_pos[2]],
                        linestyle=line_style, color=line_color[1], alpha=alpha,
                        marker=marker_style, markersize=marker_size, markerfacecolor=marker_color[1])
                

 
                
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    center_x = (np.max(positions[..., 0]) + np.min(positions[..., 0])) / 2
    center_y = (np.max(positions[..., 1]) + np.min(positions[..., 1])) / 2

    ax.set_xlim([center_x-3, center_x+3])
    ax.set_ylim([center_y-3, center_y+3])
    ax.set_zlim([0,2])
    ax.set_aspect("equal")
