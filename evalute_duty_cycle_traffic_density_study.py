import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("duty_cycle_traffic_density_study.csv", sep=", ")

# values = "mergerate"
values = "crashrate"
data = df[["dutycycle", "num_HDV", values]].pivot(
    index="dutycycle", columns="num_HDV", values=values
)

print(data)

sns.set_theme("paper")
sns.set_palette("Paired")

# sns.set(font_scale=1.2)

ax = sns.heatmap(data, annot=True, fmt="g", cmap="viridis")
ax.set_title("Collision rate")
# ax.set_title("Merge rate")

ax.set(xlabel="Number of surrounding vehicles", ylabel="Radar dutycycle [%]")
ax.invert_yaxis()

# plt.show()
plt.savefig(f"plot_{values}_traffic_density.png", dpi=600, bbox_inches="tight")
