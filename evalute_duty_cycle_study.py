import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# df = pd.read_csv("duty_cycle_study_new.csv", sep=",")
df = pd.read_csv("duty_cycle_study_with_seed.csv", sep=",")

sns.set_theme("paper")
sns.set_palette("Paired")

sns.set(font_scale=1.2)

# g = sns.relplot(
#     data=df,
#     kind="line",
#     x="evaluated",
#     y="crashrate",
#     hue="trained",  # , size="coherence", style="choice",
#     # facet_kws=dict(sharex=False),
#     palette="Paired",
#     aspect=2,
# )
#
# sns.move_legend(g, "upper left", bbox_to_anchor=(0.1, 0.95))
# g.set_axis_labels("Evaluated Duty cycle [%]", "Crash Rate")
# g._legend.set_title("Trained on duty cycle [%]")

g = sns.relplot(
    data=df,
    kind="line",
    errorbar=("ci", 95),  # plot the 95% confidence interval
    x="evaluated",
    y="ego speed",
    hue="trained",  # , size="coherence", style="choice",
    # facet_kws=dict(sharex=False),
    palette="Paired",
    aspect=2,
)

# sns.move_legend(g, "upper left", bbox_to_anchor=(0.12, 0.6))
# sns.move_legend(g, "upper left", bbox_to_anchor=(0.12, 0.95))
sns.move_legend(g, "upper left", bbox_to_anchor=(0.58, 0.7))
# g.set_axis_labels("Evaluated Duty cycle [%]", "Merge rate")
# g.set_axis_labels("Evaluated Duty cycle [%]", "Collision rate")
g.set_axis_labels("Evaluated Duty cycle [%]", "Agent velocity")
g._legend.set_title("Trained on duty cycle [%]")


# plt.show()
plt.savefig(
    # "plot_collision_rate_with_zero_with_confidence.png", dpi=600, bbox_inches="tight"
    # "plot_merge_rate_with_zero_with_confidence.png",
    "plot_speed_with_zero_with_confidence.png",
    dpi=600,
    bbox_inches="tight",
)
