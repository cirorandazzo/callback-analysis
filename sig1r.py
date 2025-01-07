# %% sig1r.py
#
# Analysis code for sig1r experiments.
# 2025.01.06 CDR

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 400

from utils.plot import (
    plot_callback_raster,
    plot_callback_raster_multiblock,
    plot_callback_raster_multiday,
    plot_callback_heatmap,
    plot_group_hist,
)

# %% AUTORELOAD

# %load_ext autoreload
# %autoreload 1
# %aimport utils.plot

# %% LOAD DFS

# NOTE: run make-df externally & save pickle outputs to pickle_folder.
pickle_folder = Path(r"F:\Sig1R-labels\processed-df\bubu")


def load_df(filename):
    with open(filename, "rb") as f:
        df = pickle.load(f)["df"]

    return df


# array of dfs. 1 per bird
pickled_dfs = [load_df(f) for f in list(pickle_folder.glob("*.pickle"))]

pickled_dfs

# %% PLOTTING SETTINGS

# family (for plot folder)
family = "bubu"

# for legend/titles: bird (condition)
condition = {
    "bu86bu36": "NE100",
    "bu88bu38": "saline",
}


def get_birdname(df):
    birdname = np.unique(df.index.get_level_values("birdname"))
    assert len(birdname) == 1  # only one bird per df
    return birdname[0]


# %% MAKE AGG DFS

# get summary stat df by day/block
dfs_by_day_block = [
    df.groupby(level=["birdname", "day", "block"]).agg(
        median_latency_s=("latency_s", lambda x: np.nanmedian(x)),
        mean_n_calls_excl_zero=("n_calls", lambda x: np.sum(x) / np.count_nonzero(x)),
        pct_trials_responded=("n_calls", lambda x: np.count_nonzero(x) / len(x)),
    )
    for df in pickled_dfs
]

max_blocks_per_day = 4
# average across blocks for each day
dfs_by_day = [
    df_blocked.loc[df_blocked.index.get_level_values("block") <= max_blocks_per_day]
    .groupby(level=["birdname", "day"])
    .agg("mean")
    for df_blocked in dfs_by_day_block
]

# %% AGG PLOT ARGUMENTS

# make sure these field_names are defined in the groupby agg above
field_names = [
    "median_latency_s",
    "mean_n_calls_excl_zero",
    "pct_trials_responded",
]
cmaps = [
    "viridis_r",
    "Purples",
    "Blues",
]
measure_ranges = [
    (0, 0.4),
    (0, 10),
    (0.8, 1),
]  # None for default colormap scale

# %% PLOT RASTERS

figsize_all_days = (4, 6)
figsize_one_day = (4, 4)

xlim = [0, 1]

raster_folder = Path(f"./data/sig1r/{family}/rasters")
tag = "-1s"  # adds label in filename.

for df in pickled_dfs:
    birdname = get_birdname(df)

    # 1 plot containing all days & all blocks
    fig, ax_all = plt.subplots(figsize=figsize_all_days)

    plot_callback_raster_multiday(df.xs(birdname), ax=ax_all)
    ax_all.get_legend().remove()
    ax_all.set(
        xlim=xlim,
        ylim=(0, len(df)),
        title=f"{birdname} ({condition[birdname]})",
    )

    fig.tight_layout()
    fig.savefig(raster_folder.joinpath(f"{birdname}{tag}-ALL.svg"))
    plt.close(fig)

    # plot all blocks per day
    days = np.unique(df.index.get_level_values("day"))
    for day in days:
        df_day = df.xs((birdname, day))

        fig, ax_day = plt.subplots(figsize=figsize_one_day)
        plot_callback_raster_multiblock(df_day, ax=ax_day)

        ax_day.set(
            xlim=xlim,
            ylim=(0, len(df_day)),
            title=f"{birdname} ({condition[birdname]}): Day {day}",
        )

        fig.tight_layout()
        fig.savefig(raster_folder.joinpath(f"{birdname}{tag}-d{day}.svg"))
        plt.close(fig)

# %% PLOT HEATMAPS

heatmap_folder = Path(f"./data/sig1r/{family}/heatmaps")

for df_blocked in dfs_by_day_block:
    birdname = get_birdname(df_blocked)

    for field_name, cmap_name, vrange in zip(field_names, cmaps, measure_ranges):
        fig, ax = plt.subplots(figsize=(9, 12))
        fig, ax, im, cbar = plot_callback_heatmap(
            df_blocked,
            field_name,
            fig=fig,
            norm=vrange,
            cmap_name=cmap_name,
        )
        ax.set(
            xlabel="Day",
            ylabel="Block",
            title=f"{birdname} ({condition[birdname]}): {field_name}",
        )

        fig.savefig(heatmap_folder.joinpath(f"{birdname}-heatmap-{field_name}.svg"))
        plt.close()

# %% LINE PLOTS: MEAN

lineplot_folder = Path(f"./data/sig1r/{family}/lineplots")

for field_name, vrange in zip(field_names, measure_ranges):
    fig, ax = plt.subplots(figsize=(12, 9))

    for df in dfs_by_day:
        birdname = get_birdname(df)
        ax.plot(
            df.index.get_level_values("day"),
            df[field_name],
            label=f"{birdname} ({condition[birdname]})",
        )

    ax.set(
        xlabel="Day",
        ylabel=field_name,
        title=f"{field_name}, mean of first {max_blocks_per_day + 1} blocks/ day"
    )

    ax.set_xticks(df.index.get_level_values("day"))
    ax.legend()

    fig.savefig(lineplot_folder.joinpath(f"lineplot-{field_name}.svg"))
    plt.close()

# %% D1 vs D5 distribution

distribution_folder = Path(f"./data/sig1r/{family}/distributions")

field_name = "latency_s"
binwidth = 0.02
suptitle = f"{field_name}: all trials in first 5 blocks, d1 v. d5"
xlabel = "Latency to first call (s)"

fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True)

for df, ax in zip(pickled_dfs, axs):
    birdname = get_birdname(df)
    plot_group_hist(
        df.loc[df.index.get_level_values("block") <= max_blocks_per_day],
        field=field_name,
        grouping_level="day",
        groups_to_plot=(1, 5),
        ax=ax,
        binwidth=binwidth,
        stair_kwargs={
            "fill": True,
            "alpha": 0.7,
        },
    )

    ax.set(title=f"{birdname} ({condition[birdname]})")

fig.suptitle(suptitle)
axs[-1].set(xlabel=xlabel)

fig.savefig(distribution_folder.joinpath(f"distribution-{field_name}.svg"))
plt.close()
