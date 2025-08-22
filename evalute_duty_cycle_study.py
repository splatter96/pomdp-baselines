import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("duty_cycle_study.csv", sep=",")

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
    x="evaluated",
    y="mergerate",
    hue="trained",  # , size="coherence", style="choice",
    # facet_kws=dict(sharex=False),
    palette="Paired",
    aspect=2,
)

sns.move_legend(g, "upper left", bbox_to_anchor=(0.12, 0.55))
g.set_axis_labels("Evaluated Duty cycle [%]", "Merge Rate")
g._legend.set_title("Trained on duty cycle [%]")


# plt.show()
plt.savefig("plot_mergerate.png", dpi=600, bbox_inches="tight")
