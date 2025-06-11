import glob
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
# import plotly.express as px

matplotlib.rcParams.update({"font.size": 12})

to_step = 10000


def rle(inarray):
    """run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    returns: tuple (runlengths, startpositions, values)"""
    ia = np.asarray(inarray)  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return (z, p, ia[i])  # run lengths, positions in array, value of run


# for file in glob.glob("radars*_probability*_60_frametime.npy"):
# for file in glob.glob("radars_15*_any*_60_frametime.npy"):
# for file in glob.glob("radars_14_2000_any_new_auto_60_frametime_new.npy"):
for file in glob.glob("radars_50_2000_any_new_auto_60_frametime_new.npy"):
    with open(file, "rb") as f:
        data = np.load(f)

    print(data.shape)

    fig, axs = plt.subplots(4)
    fig.tight_layout()
    fig.suptitle(file)
    for i in range(4):
        axs[i].plot(data[:to_step, i], label=f"radar {i}")
        axs[i].legend(loc="upper left")
        axs[i].set_ylim(-0.1, 1.1)

        # compute average run lengths
        z, _, val = rle(data[:to_step, i])

        # get only the runs where interference happens
        z = z[val == True]
        print(f"{file} {i}: {np.average(z)}")

    plt.show()
    # plt.savefig(f"plot_{file}_60_frametime.png", dpi=900)
    plt.close()
