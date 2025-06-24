import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("duty_cycle_study.csv", sep=",")

print(df)


g = sns.relplot(
    data=df,
    kind="line",
    x="evaluated",
    y="crashrate",
    hue="trained",  # , size="coherence", style="choice",
    # facet_kws=dict(sharex=False),
)

g.set_axis_labels("Evaluated Duty cycle [%]", "Crash Rate")
g._legend.set_title("Trained on duty cycle [%]")

g = sns.relplot(
    data=df,
    kind="line",
    x="evaluated",
    y="mergerate",
    hue="trained",  # , size="coherence", style="choice",
    # facet_kws=dict(sharex=False),
)

g.set_axis_labels("Evaluated Duty cycle [%]", "Merge Rate")
g._legend.set_title("Trained on duty cycle [%]")


plt.show()
