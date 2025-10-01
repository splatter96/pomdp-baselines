import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# plt.rcParams["text.usetex"] = True

df = pd.read_csv("right_bias_study.csv", sep=",")

sns.set_theme("paper")
sns.set_palette("Paired")

sns.set(font_scale=1.2)

fig, ax = plt.subplots(figsize=(12, 4))
fig.tight_layout()

g = sns.lineplot(
    data=df,
    ax=ax,
    x="right_bias",
    y="crashrate",
    # y="mergerate",
)

ax.set(ylim=(0, 0.043))
# ax.set(xlabel="Right bias [$ms^{-2}$]", ylabel="Collision rate")
# ax.set(xlabel="Right bias [$ms^{-2}$]", ylabel="Merge rate")
ax.set(xlabel="Right bias [m/s^2]", ylabel="Collision rate")
# ax.set(xlabel="Right bias [m/s^2]", ylabel="Merge rate")

plt.yticks(np.arange(0, 0.05, 0.01))

# plt.show()
plt.savefig("plot_collision_rate_right_bias.png", dpi=600, bbox_inches="tight")
# plt.savefig("plot_merge_rate_right_bias.png", dpi=600, bbox_inches="tight")
