# %% raster-example.py
#
# Example usage of callback raster plotting functions.
# 2025.01.06 CDR

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 400

# %%

# %load_ext autoreload
# %autoreload 1
# %aimport utils.plot

from utils.plot import plot_callback_raster, plot_callback_raster_multiblock, plot_callback_raster_multiday

# %% run make-df externally.

# %%

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

# %% raster plots

# %%

df = pickled_dfs[0]


# %% plot one block

# bird, day, block
index = ("bu86bu36", 1, 1)

ax_block = plot_callback_raster(
    df.xs(index),
    plot_stim_blocks=True,
    show_legend=True,
    y_offset=0,
    call_types_to_plot = "all",
    force_yticks_int=True,
)

ax_block.set(title="%s: Day %s, Block %s" % index)

plt.show()

# %% plot one day, all blocks

index = ("bu86bu36", 1)
ax_day = plot_callback_raster_multiblock(df.xs(index))

ax_day.set(title="%s: Day %s, All Blocks" % index)

# %% plot all days

index = "bu86bu36"

ax_all = plot_callback_raster_multiday(df.xs(index))
ax_all.get_legend().remove()
ax_all.set(
    ylim=(0, len(df)),
    title=index,
)