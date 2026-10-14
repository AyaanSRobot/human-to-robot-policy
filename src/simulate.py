import mujoco
import numpy as np
import pandas as pd
import imageio.v2 as imageio
from pathlib import Path

# Adjust these to one clear transport segment
START_S = 100.0
END_S = 110.0

MODEL_PATH = "../../mujoco_menagerie/franka_emika_panda/scene.xml"
CSV_PATH = "grasp_trajectory.csv"

OUTPUT = Path("results/panda_demo.mp4")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# 1. Load one recorded trajectory
df = pd.read_csv(CSV_PATH)

demo = df[(df.timestamp_ms >= START_S * 1000) & (df.timestamp_ms <= END_S * 1000)].copy()

# Do not bridge large missing-detection gaps
demo = demo.dropna(subset=["x", "y"])

if len(demo) < 3:
    raise ValueError("Not enough tracked points in this segment")

# Normalize time and horizontal displacement
time = demo.timestamp_ms.to_numpy() / 1000
time -= time[0] # normalizing for trajectory to start at t=0

xy = demo[["x", "y"]].to_numpy(copy=True)
xy -= xy[0].copy()

# Choose an approximate displacement scale initially
scale = 0.25  # metres per normalized image unit

# Synthetic lift trajectory (maximum 4 cm)
phase = time / time[-1]
lift = 0.04 * np.sin(np.pi * phase) # simulating up and down lift motion

# 2. Load Panda
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

if model.nkey:
    mujoco.mj_resetDataKeyframe(model, data, 0)

mujoco.mj_forward(model, data)

hand_id = model.body("hand").id
p0 = data.xpos[hand_id].copy()

# 3. Construct desired Cartesian path
targets = np.column_stack([
    p0[0] + scale * xy[:, 0], # x
    p0[1] - scale * xy[:, 1], # y
    p0[2] + lift              # z
])

# 4. Solve IK for every point
def solve_ik(target):
    damping = 0.05

    for _ in range(100):
        mujoco.mj_forward(model, data)

        error = target - data.xpos[hand_id]

        if np.linalg.norm(error) < 0.002:
            return True

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacBody(
            model, data, jacp, jacr, hand_id
        )

        J = jacp[:, :7]

        dq = J.T @ np.linalg.solve(
            J @ J.T + damping**2 * np.eye(3),
            error
        )

        data.qpos[:7] += 0.5 * dq

        for j in range(7):
            joint = model.joint(f"joint{j+1}")

            if joint.limited[0]:
                low, high = joint.range
                data.qpos[j] = np.clip(data.qpos[j], low, high) # clip to joint range

    mujoco.mj_forward(model, data)
    return np.linalg.norm(target - data.xpos[hand_id]) < 0.002 # true or false

# 5. Render and save video
frames = []
failures = 0
errors = []

with mujoco.Renderer(
    model, height=480, width=640
) as renderer:

    for target in targets:
        success = solve_ik(target)
        failures += not success

        mujoco.mj_forward(model, data)

        error = np.linalg.norm(
            target - data.xpos[hand_id]
        )
        errors.append(error)

        renderer.update_scene(data)
        frames.append(renderer.render().copy())

imageio.mimsave(
    OUTPUT,
    frames,
    fps=10,
    macro_block_size=16
)

print(f"Video saved to: {OUTPUT}")
print(f"IK failures: {failures}/{len(targets)}")
print(f"Mean position error: {np.mean(errors):.5f} m")
print(f"Maximum position error: {np.max(errors):.5f} m")
