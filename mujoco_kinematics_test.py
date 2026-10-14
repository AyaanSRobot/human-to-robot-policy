import numpy as np
import mujoco
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

model_path = (
    project_root.parent
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Initialize the robot
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)

mujoco.mj_forward(model, data) # using mujoco's forward kinemaics (fk) to find gripper

hand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)

current_position = data.xpos[hand_id].copy()

target_position = current_position + np.array([
    0.05, 0.0, 0.0
]) # target position 5 cm away

print("Current position:", current_position)
print("Target position:", target_position)

# move end effector to target position
damping = 0.05
step_size = 0.5

for iteration in range(200):

    mujoco.mj_forward(model, data)

    current_position = data.xpos[hand_id].copy()
    error = target_position - current_position

    if np.linalg.norm(error) < 0.001:
        print("Target reached!")
        break

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacBody(
        model, data, jacp, jacr, hand_id
    )

    # Use only the 7 arm joints
    J = jacp[:, :7]

    dq = J.T @ np.linalg.solve( # compute joint positions using damped least-squares inverse kinematics
        J @ J.T + damping**2 * np.eye(3),
        error
    )

    # Kinematic update only
    data.qpos[:7] += step_size * dq

    # Keep joint angles within limits
    for j in range(7):
        joint = model.joint(f"joint{j+1}")
        if joint.limited[0]:
            low, high = joint.range
            data.qpos[j] = np.clip(data.qpos[j], low, high)

mujoco.mj_forward(model, data)

print("Final position:", data.xpos[hand_id])
print("Position error:", np.linalg.norm(
    target_position - data.xpos[hand_id]
))
# final position error of 0.546 mm
# made using aid of AI