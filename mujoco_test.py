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

#AI test script