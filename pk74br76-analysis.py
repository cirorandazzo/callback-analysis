# %% pk74br76-analysis.py
#
# Analysis code for analysis of pk74br76 hvc pharmacology experiments.
# 2025.01.09 CDR

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

# %%
# AUTORELOAD

# %load_ext autoreload
# %autoreload 1
# %aimport utils.plot

# %% LOAD DFS

# NOTE: run make-df externally & save pickle outputs to pickle_folder.
filename = Path(r".\data\hvc_pharm\pk74br76.pickle")
birdname = "pk74br76"

savefig_root = Path(r"./data/hvc_pharm/pk74br76")

exclude_song = False

with open(filename, "rb") as f:
    data = pickle.load(f)
    df = data["df"]
    calls_df = data["calls_all"]

if exclude_song:
    has_song = df["call_types"].apply(lambda x: "Song" in x)
    print(f"Removing {sum(has_song)} trials with Song.")
    df = df.loc[~has_song]

df

# %%
# PLOTTING SETTINGS

# names courtesy of coolors.co
colors_drug = {
    "saline": "#054A91",  # "Polynesian blue"
    "muscimol_0_1mM": "#FE5F55",  # "Bittersweet"
    "muscimol_0_1mM_washout": "#FFAFA9",  # "Melon"
    "gabazine_0_1mM": "#D1BCE3",  # "Thistle"
    "baclofen_0_1mM": "#DF9A5E",  # "Buff"
    "saclofen_0_1mM": "#B5A886",  # "Khaki"
}

drug_days = {}
for day in df.index.unique(level="day"):
    drug = df.xs(key=day, level="day").index.unique(level="drug")
    assert len(drug) == 1, f">1 drug found on day {day}"
    drug_days[day] = drug[0]

drug_days

# %% MAKE AGG DFS

agg_functions = dict(
    median_latency_s=("latency_s", lambda x: np.nanmedian(x)),
    mean_n_calls_excl_zero=("n_calls", lambda x: np.sum(x) / np.count_nonzero(x)),
    pct_trials_responded=("n_calls", lambda x: np.count_nonzero(x) / len(x)),
)

# get summary stat df by day/block
df_by_drug_block = df.groupby(level=["birdname", "day", "block"]).agg(**agg_functions)

# average across blocks for each day
df_by_day = df.groupby(level=["birdname", "day"]).agg(**agg_functions)

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
# measure_ranges = [
#     (0, 0.4),
#     (0, 10),
#     (0.8, 1),
# ]  # None for default colormap scale
measure_ranges = [None] * 3
# %% PLOT RASTERS

figsize_all_days = (5, 10)
figsize_one_day = (5, 5)

raster_folder = savefig_root.joinpath(r"rasters")

xlim = [0, 3]
tag = ""

# xlim = [0, 1]
# tag = "-1s"  # adds label in filename.

# 1 plot containing all days & all blocks
fig, ax_all = plt.subplots(figsize=figsize_all_days)

plot_callback_raster_multiday(
    df.xs(birdname).set_index("raster_timepoint", append=True),
    ax=ax_all,
    day_labels=drug_days,
    subday_level="raster_timepoint",
    hline_block_kwargs = dict(
        colors="k",
        linestyles="dashed",
        linewidths=0.2,
    )
)
ax_all.get_legend().remove()
ax_all.set(
    xlim=xlim,
    ylim=(0, len(df)),
    title=f"{birdname}",
)

fig.tight_layout()
fig.savefig(raster_folder.joinpath(f"{birdname}{tag}-ALL.svg"))
plt.close(fig)

# %%
# plot all blocks per day
days = np.unique(df.index.get_level_values("day"))
for day in days:
    df_day = df.xs((birdname, day))

    fig, ax_day = plt.subplots(figsize=figsize_one_day)
    plot_callback_raster_multiblock(
        df_day.set_index("raster_timepoint", append=True),
        ax=ax_day,
        block_level_name="raster_timepoint",
        show_block_axis=False,
    )

    ax_day.set(
        xlim=xlim,
        ylim=(0, len(df_day)),
        title=f"{birdname}: {drug_days[day]} (d{int(day)})",
    )

    fig.tight_layout()
    fig.savefig(raster_folder.joinpath(f"{birdname}{tag}-{drug_days[day]}-d{int(day)}.svg"))
    plt.close(fig)

# %% PLOT HEATMAPS

heatmap_folder = savefig_root.joinpath(r"heatmaps")

for field_name, cmap_name, vrange in zip(field_names, cmaps, measure_ranges):
    fig, ax = plt.subplots(figsize=(9, 12))
    fig, ax, im, cbar = plot_callback_heatmap(
        df_by_drug_block,
        field_name,
        fig=fig,
        norm=vrange,
        cmap_name=cmap_name,
        day_labels=drug_days,
    )
    ax.set(
        ylabel="Block",
        title=f"{birdname}: {field_name}",
    )
    ax.tick_params(axis='x', labelrotation=45)

    fig.savefig(heatmap_folder.joinpath(f"{birdname}-heatmap-{field_name}.svg"))
    plt.close()


# %% LINEPLOTS BY DAY

lineplot_folder = savefig_root.joinpath(r"lineplots")

for field_name, vrange in zip(field_names, measure_ranges):
    fig, ax = plt.subplots(figsize=(12, 9))

    days = df_by_day.index.get_level_values("day")

    ax.plot(
        days,
        df_by_day[field_name],
        label=f"{birdname}",
    )

    xrange = 0.2 * (max(days) - min(days))

    ax.set(
        # xlabel="Day",
        ylabel=field_name,
        title=f"{field_name}",
        xlim=[min(days) - xrange, max(days) + xrange],  # normalize to middle 3rd
    )

    ax.set_xticks(ticks=days, labels=[drug_days[d] for d in days])
    ax.tick_params(axis='x', labelrotation=45)
    ax.legend()

    fig.tight_layout()

    fig.savefig(lineplot_folder.joinpath(f"lineplot-{field_name}.svg"))
    plt.close()

# %% distribution (latency, n_calls)

distribution_folder = savefig_root.joinpath(r"distributions")


def distribution_plot(
    dfs_array,
    field_name,
    binwidth,
    suptitle,
    xlabel,
    grouping_level="drug",
    key_groupings=[["saline", "muscimol_0_1mM"]],
    **plot_group_hist_kwargs,
):

    for keys in key_groupings:
        d_label = "__".join(keys)
        fig, axs = plt.subplots(nrows=len(dfs_array), sharex=True, sharey=True)

        # in case of just 1
        try:
            iter(axs)
        except TypeError:
            axs = [axs]

        for df, ax in zip(dfs_array, axs):
            plot_group_hist(
                df,
                field=field_name,
                grouping_level=grouping_level,
                groups_to_plot=keys,
                ax=ax,
                binwidth=binwidth,
                stair_kwargs={
                    "fill": True,
                    "alpha": 0.7,
                },
                **plot_group_hist_kwargs,
            )

            ax.set(title=f"{birdname}")

        fig.suptitle(suptitle)
        axs[-1].set(xlabel=xlabel)

        fig.savefig(
            distribution_folder.joinpath(f"distribution-{field_name}-{d_label}.svg")
        )
        plt.close()


to_plot = [
    dict(
        dfs_array=[df],
        group_colors=colors_drug,
        field_name="latency_s",
        binwidth=0.04,
        suptitle=f"Latency",
        xlabel="Latency to first call (s)",
    ),
    dict(
        dfs_array=[df],
        group_colors=colors_drug,
        field_name="n_calls",
        binwidth=1,
        suptitle=f"# Calls / Trial",
        xlabel="# Calls / Trial",
    ),
    dict(
        dfs_array=[calls_df],
        group_colors=colors_drug,
        field_name="ici",
        binwidth=0.04,
        suptitle=f"ICI",
        xlabel="ICI (s)",
        ignore_nan=True,
    ),
]

[distribution_plot(**d) for d in to_plot]
