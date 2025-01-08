# %% sig1r.py
#
# Analysis code for sig1r experiments.
# 2025.01.06 CDR

import os
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
exclude_song = True


def load_df(filename, df_key="df", exclude_song=False):
    with open(filename, "rb") as f:
        df = pickle.load(f)[df_key]

    if exclude_song:
        has_song = df["call_types"].apply(lambda x: "Song" in x)
        print(f"Removing {sum(has_song)} trials with Song.")
        df = df.loc[~has_song]

    return df


files = list(pickle_folder.glob("*.pickle"))

# array of dfs, 1 per bird. main dfs, where each row is a trial.
pickled_dfs = [
    load_df(
        f,
        df_key="df",
        exclude_song=exclude_song,
    )
    for f in files
]

# array of dfs, 1 per bird. aux dfs, where each row is a call (or stim, syllable, etc.)
calls_dfs = [
    load_df(
        f,
        df_key="calls_all",
        exclude_song=False,  # doesn't work for this df format
    )
    for f in files
]

pickled_dfs

# %% PLOTTING SETTINGS

# figure savefolder root
savefig_root = Path(r"./data/sig1r/bubu/song_excluded")

# for legend/titles: bird (condition)
condition = {
    "bu86bu36": "NE100",
    "bu88bu38": "saline",
}


def get_birdname(df):
    birdname = np.unique(df.index.get_level_values("birdname"))
    assert len(birdname) == 1  # only one bird per df
    return birdname[0]


# prepare plot subfolders
subfolders = ["rasters", "heatmaps", "lineplots", "distributions"]

# for sf in subfolders:
#     os.makedirs(savefig_root.joinpath(sf))

# %% MAKE AGG DFS

agg_functions = dict(
    median_latency_s=("latency_s", lambda x: np.nanmedian(x)),
    mean_n_calls_excl_zero=("n_calls", lambda x: np.sum(x) / np.count_nonzero(x)),
    pct_trials_responded=("n_calls", lambda x: np.count_nonzero(x) / len(x)),
)

# get summary stat df by day/block
dfs_by_day_block = [
    df.groupby(level=["birdname", "day", "block"]).agg(
        **agg_functions
    )
    for df in pickled_dfs
]

max_blocks_per_day = 4
# average across blocks for each day
dfs_by_day = [
    df.groupby(level=["birdname", "day"]).agg(
        **agg_functions
    )
    for df in pickled_dfs
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

raster_folder = savefig_root.joinpath(r"rasters")

xlim = [0, 3]
tag = ""

# xlim = [0, 1]
# tag = "-1s"  # adds label in filename.

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

heatmap_folder = savefig_root.joinpath(r"heatmaps")

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

# %% LINEPLOTS BY DAY

lineplot_folder = savefig_root.joinpath(r"lineplots")

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
        title=f"{field_name} (first {max_blocks_per_day + 1} blocks/ day)",
    )

    ax.set_xticks(df.index.get_level_values("day"))
    ax.legend()

    fig.savefig(lineplot_folder.joinpath(f"lineplot-{field_name}.svg"))
    plt.close()

# %% D1 vs D5 distribution (latency, n_calls)

distribution_folder = savefig_root.joinpath(r"distributions")


def distribution_plot(
    dfs_array,
    field_name,
    binwidth,
    suptitle,
    xlabel,
    day_groupings=((1, 3), (1, 5), (3, 5), (1, 3, 5)),
    **plot_group_hist_kwargs,
):

    for days in day_groupings:
        d_label = "d" + "_".join([str(i) for i in days])
        fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True)

        for df, ax in zip(dfs_array, axs):
            birdname = get_birdname(df)
            plot_group_hist(
                df.loc[df.index.get_level_values("block") <= max_blocks_per_day],
                field=field_name,
                grouping_level="day",
                groups_to_plot=days,
                ax=ax,
                binwidth=binwidth,
                stair_kwargs={
                    "fill": True,
                    "alpha": 0.7,
                },
                **plot_group_hist_kwargs,
            )

            ax.set(title=f"{birdname} ({condition[birdname]})")

        fig.suptitle(suptitle)
        axs[-1].set(xlabel=xlabel)

        fig.savefig(
            distribution_folder.joinpath(f"distribution-{field_name}-{d_label}.svg")
        )
        plt.close()


to_plot = [
    dict(
        dfs_array=pickled_dfs,
        field_name="latency_s",
        binwidth=0.04,
        suptitle=f"Latency: all trials in first {max_blocks_per_day + 1} blocks",
        xlabel="Latency to first call (s)",
    ),
    dict(
        dfs_array=pickled_dfs,
        field_name="n_calls",
        binwidth=1,
        suptitle=f"# Calls / Trial: all trials in first {max_blocks_per_day + 1} blocks",
        xlabel="# Calls / Trial",
    ),
    dict(
        dfs_array=calls_dfs,
        field_name="ici",
        binwidth=0.04,
        suptitle=f"ICI: all trials in first {max_blocks_per_day + 1} blocks",
        xlabel="ICI (s)",
        ignore_nan=True,
    ),
]

[distribution_plot(**d) for d in to_plot]
