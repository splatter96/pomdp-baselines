import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# plt.rcParams["text.usetex"] = True

df = pd.read_csv("merge_length_study_with_seed.csv", sep=",")
print(df.columns)

sns.set_theme("paper")
sns.set_palette("Paired")

sns.set(font_scale=1.2)

fig, ax = plt.subplots(figsize=(12, 4))
fig.tight_layout()

g = sns.lineplot(
    data=df,
    ax=ax,
    x="merge_length",
    # y="crashrate",
    y="mergerate",
)

# ax.set(ylim=(0, 0.04))
# ax.set(xlabel="Merge lane length [m]", ylabel="Collision rate")
ax.set(xlabel="Merge lane length [m]", ylabel="Merge rate")

# plt.show()
# plt.savefig("plot_collision_rate_merge_length.png", dpi=600, bbox_inches="tight")
plt.savefig("plot_merge_rate_merge_length.png", dpi=600, bbox_inches="tight")
