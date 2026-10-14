from pathlib import Path
import mujoco

project_root = Path(__file__).resolve().parents[1]

model_path = (
    project_root.parent
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

print("Panda model loaded successfully!")

print("\nJoints:")
for i in range(model.njnt):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))

print("\nActuators:")
for i in range(model.nu):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))

print("\nBodies:")
for i in range(model.nbody):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))

#AI test script