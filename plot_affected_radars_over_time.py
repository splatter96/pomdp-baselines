import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px

with open("radars_40_6000.npy", "rb") as f:
    data = np.load(f)

print(data[:, 0].shape)

plt.plot(data[:, 0], label="radar 0")
plt.plot(data[:, 1], label="radar 1")
# plt.plot(data[:, 2], label="radar 2")
# plt.plot(data[:, 3], label="radar 3")
plt.show()
