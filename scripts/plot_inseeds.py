# %%
import pandas as pd
import matplotlib.pyplot as plt


output_path = "./simulations/output/coupled_test/"

all_output = pd.read_csv(f"{output_path}/inseeds.csv")
ts = pd.pivot_table(
    all_output, values="value", index=["year", "cell"], columns=["variable"]
)

ts.query("cell == 0")
ts_mean = ts.groupby("year").mean()

ts_mean = ts_mean.drop("average harvest date", axis=1)

# create a figure with multiple subplots
fig, axes = plt.subplots(
    nrows=len(ts_mean.columns), ncols=1, sharex=True, figsize=(8, 10)
)

# plot each variable on its own subplot
for i, col in enumerate(ts_mean.columns):
    ts_mean[col].plot(ax=axes[i], label=col)
    axes[i].set_ylabel("Mean over cells")
    axes[i].legend()

# set the x-axis label
plt.xlabel("Year")

# set the plot title
plt.suptitle("Netherlands: Spreading effects of (no-)till practices")

# show the plot
plt.show()

# fig.savefig("test.pdf")

# %%
