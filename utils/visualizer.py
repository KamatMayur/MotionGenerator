import numpy as np
from pythreejs import *
from IPython.display import display

# Example data
num_frames = 100
num_joints = 15
animation = np.random.rand(num_frames, num_joints, 3) * 100  # Random example data
parent_indices = np.random.randint(-1, num_joints, size=num_joints)  # Random parent indices

# Create a basic scene
scene = Scene()
camera = PerspectiveCamera(position=[0, 0, 150], up=[0, 1, 0], aspect=1.6)
renderer = Renderer(scene=scene, camera=camera, background='white', width=800, height=600)

# Initialize joints and lines
joints = [Mesh(geometry=SphereGeometry(radius=2, widthSegments=8, heightSegments=8), material=MeshBasicMaterial(color='red')) for _ in range(num_joints)]
lines = [Line(geometry=BufferGeometry(attributes={'position': BufferAttribute(np.array([]).astype(np.float32).reshape(0, 3), normalized=False)}), material=LineBasicMaterial(color='black')) for _ in range(num_joints)]

for joint in joints:
    scene.add(joint)

for line in lines:
    scene.add(line)

def update_frame(frame_index):
    for i, joint in enumerate(joints):
        joint.position = animation[frame_index, i].tolist()
    
    for i, line in enumerate(lines):
        if parent_indices[i] >= 0:
            start = animation[frame_index, i]
            end = animation[frame_index, parent_indices[i]]
            line.geometry.attributes['position'].array = np.array([start, end], dtype=np.float32).flatten()
            line.geometry.attributes['position'].needsUpdate = True

def animate():
    frame_index = 0
    while True:
        update_frame(frame_index)
        renderer.render(scene, camera)
        frame_index = (frame_index + 1) % num_frames

# Display the scene
display(renderer)

# Start animation
animate()