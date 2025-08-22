import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("duty_cycle_traffic_density_study.csv", sep=", ")

values = "mergerate"
data = df[["dutycycle", "num_HDV", values]].pivot(
    index="dutycycle", columns="num_HDV", values=values
)

print(data)

sns.set_theme("paper")
sns.set_palette("Paired")

sns.set(font_scale=1.2)

sns.heatmap(data, annot=True, fmt="g", cmap="viridis")

# sns.move_legend(g, "upper left", bbox_to_anchor=(0.12, 0.55))
# g.set_axis_labels("Evaluated Duty cycle [%]", "Merge Rate")
# g._legend.set_title("Trained on duty cycle [%]")


plt.show()
# plt.savefig("plot_mergerate.png", dpi=600, bbox_inches="tight")
