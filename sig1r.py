# %% sig1r.py
#
# Analysis code for sig1r experiments.
# 2025.01.06 CDR

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 400

from utils.plot import plot_callback_raster, plot_callback_raster_multiblock, plot_callback_raster_multiday

# %%

# run make-df externally & save pickle outputs to pickle_folder.
pickle_folder = Path(r"F:\Sig1R-labels\processed-df\bubu")

def load_df(filename):
    with open(filename, 'rb') as f: 
        df = pickle.load(f)["df"]
    
    return df

pickled_dfs = [
    load_df(f)
    for f in list( pickle_folder.glob("*.pickle"))
]

pickled_dfs

# %%

condition = {
    "bu86bu36": "NE100",
    "bu88bu38": "saline",
}

# %%

figsize_all_days = (4,6)
figsize_one_day = (4,4)

xlim = [0, 1]

raster_folder = Path(f"./data/sig1r/rasters")
tag = "-1s"

for df in pickled_dfs:

    birdname = np.unique(df.index.get_level_values("birdname"))
    assert len(birdname) == 1  # only one bird per df
    birdname = birdname[0]

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
            title=f"{birdname} ({condition[birdname]}): Day {day}"
        )

        fig.tight_layout()
        fig.savefig(raster_folder.joinpath(f"{birdname}{tag}-d{day}.svg"))
        plt.close(fig)
