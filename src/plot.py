import matplotlib.pyplot as plt 
import pandas as pd

# plot hand trajectory

msft = pd.read_csv('grasp_trajectory.csv')
msft.plot(x="timestamp_ms",y=["x", "y"],figsize=(12, 5),
    title="Hand Grasp Trajectory")

plt.savefig("results/hand_trajectory.png", dpi=300, bbox_inches="tight")
