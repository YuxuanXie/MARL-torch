from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
import pandas as pd

name = "tiger"

data = pd.read_csv("{}.csv".format(name))

rc("text", usetex = True)
rc("font", size = 30)
fig, ax = plt.subplots()

ax.plot(data["eposides"], data["loss"], color = "r")

ax.set_ylabel(r"$loss$", labelpad = 30)
ax.set_xlabel(r"$episode$", labelpad = 30)
ax.set_title("$\mathtt{{{}}}$".format(name), pad = 20)
ax.axvline(x=50000)
fig.set_tight_layout(True)
fig.set_size_inches(16,9)
    
# fig.savefig("results/rl/{}/coop_asym_qlearning/{}m{:d}n{:d}k{:d}w{:d}.png".format(name, "curse", memory, 100, 10000, 10000))
fig.savefig("{}.png".format(name))
plt.show()

