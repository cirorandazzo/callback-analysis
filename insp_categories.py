# %%
#
# insp categories
#
# idea: what's the basis set of inspiratory patterns?
# (1) look at distribution of where first insp occurs (should be random)
# (2) categories
#   (a) first pass: binned averages
#   (b) second pass: umap

import glob
import json
import os
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils.breath import get_first_breath_segment
from utils.file import parse_birdname

# %%
# parameters

fs = 44100
buffer_ms = 10

# %%
# load `all_trials` dataframe

with open("./data/breath_figs-spline_fit/all_trials.pickle", "rb") as f:
    all_trials = pickle.load(f)


all_trials.reset_index(inplace=True)

# add birdname
all_trials["birdname"] = all_trials["wav_filename"].apply(parse_birdname)


all_trials.set_index(["birdname", "wav_filename", "stims_index"], inplace=True)
all_trials.sort_index(inplace=True)

all_trials

# %%
# get first insp for all trials
#
# stim-aligned & in samples

all_trials["ii_first_insp"] = all_trials.apply(
    get_first_breath_segment,
    axis=1,
    breath_type="insp",
    fs=fs,
    buffer_s=0,
    return_stim_aligned=True,
    return_unit="samples",
)

all_trials


# %%
# reject trials without insp
#
# these occur at end of callbacks - final trial cuts off immediately post-stim

ii_no_insps = all_trials["ii_first_insp"].isna()

rejected = all_trials.loc[ii_no_insps]
all_trials = all_trials.loc[~ii_no_insps]

print("Rejected trials: no inspiration.")
rejected

# %%
# plot timing histograms
#
#

buffer_fr = buffer_ms * fs / 1000

hist_kwarg = dict(bins=30, alpha=0.5)


def plot_timing_hist(all_trials, subtitle, **hist_kwarg):
    all_insps = np.vstack(all_trials["ii_first_insp"]).T

    onsets_ms = all_insps[0, :] / fs * 1000
    offsets_ms = all_insps[1, :] / fs * 1000

    fig, ax = plt.subplots()

    ax.hist(onsets_ms, label="insp onset", **hist_kwarg)
    ax.hist(offsets_ms, label="insp offset", **hist_kwarg)
    ax.legend()

    ax.set(
        title=f"first inspiration timing: {subtitle}",
        xlabel="time, stim-aligned (ms)",
        ylabel="count",
        xlim=[-350, 900],
    )

    return fig, ax


# all birds merged
plot_timing_hist(all_trials, "all birds", **hist_kwarg)

# by bird
for birdname, all_trials_bird in all_trials.groupby(level="birdname"):
    plot_timing_hist(all_trials_bird, birdname, **hist_kwarg)
