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
import pathlib
import pickle

import numpy as np
import pandas as pd
from scipy.signal import butter

import matplotlib.pyplot as plt

import umap

plt.rcParams.update({"savefig.dpi": 400})

from utils.audio import AudioObject
from utils.breath import get_first_breath_segment
from utils.file import parse_birdname

# %%
# parameters

fs = 44100
buffer_ms = 500

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


# %%
# get window encompassing all insps

buffer_fr = int(buffer_ms * fs / 1000) + 1
all_insps = np.vstack(all_trials["ii_first_insp"]).T

window = (all_insps.min() - buffer_fr, all_insps.max() + buffer_fr)
window

# %%
# get trace for whole window

# filter
b, a = butter(N=2, Wn=50, btype="low", fs=fs)


def get_breath_for_trial(trial, breath, window, fs):

    ii_audio = np.array(window) + int(trial["trial_start_s"] * fs)

    # check if window extends past file end
    if ii_audio[1] > len(breath):
        missing_frames = ii_audio[1] - len(breath)

        cut_breath = np.pad(
            breath[ii_audio[0] :],
            [0, missing_frames],
        )

    else:
        cut_breath = breath[np.arange(*ii_audio)]

    return cut_breath


for wav_filename, file_trials in all_trials.groupby("wav_filename"):

    breath = AudioObject.from_wav(wav_filename, channels=1, b=b, a=a).audio_filt

    all_trials.loc[file_trials.index, "breath"] = file_trials.apply(
        get_breath_for_trial,
        axis=1,
        breath=breath,
        window=window,
        fs=fs,
    )

    # TODO: store file zero point & min for normalizing inspirations

all_trials


# %%
# get insp only trace (zero-padded to window size)


def get_insp_trace(trial, window, pad_value=0):

    breath = trial["breath"].copy()
    insp_on, insp_off = trial["ii_first_insp"]

    # CHECKS
    pre_insp = insp_on - window[0]
    assert pre_insp > 0, f"{window[0]} | {insp_on}"

    post_insp = insp_off - window[1]
    assert post_insp < 0, f"{window[1]} | {insp_off}"

    # DO PADDING
    breath[:pre_insp] = pad_value
    breath[post_insp:] = pad_value

    return breath


all_trials.loc[:, "insps_padded"] = all_trials.apply(
    get_insp_trace, axis=1, window=window
)

all_trials

# %%
# plot all aligned traces

fig, ax = plt.subplots()
for bin in all_trials["breath"]:
    ax.plot(np.arange(*window), bin)

ax.set(
    title="breaths",
    xlabel="samples (stim-aligned)",
    ylabel="amplitude",
)

plt.show()
# %%
# plot all aligned traces - insp only

fig, ax = plt.subplots()
for bin in all_trials["insps_padded"]:
    ax.plot(np.arange(*window), bin)

ax.set(
    title="padded insps",
    xlabel="samples (stim-aligned)",
    ylabel="amplitude",
)

plt.show()

# %%
# plot breath traces by insp bin

save_folder = pathlib.Path("./data/insp_bins-offset")


# get distr of offets
all_insps = np.vstack(all_trials["ii_first_insp"]).T
offsets_ms = all_insps[1, :] / fs * 1000


# make & plot histogram
hist, edges = np.histogram(offsets_ms, bins=40, range=(0, 840))

fig, ax = plt.subplots()

ax.stairs(hist, edges)
ax.set(
    xlabel="Insp offset (ms)",
    ylabel="Count",
    title="Insp offset distribution: all birds",
)

fig.savefig(save_folder.joinpath("offset_distr.jpg"))
plt.close(fig)


# plot breath traces by bin
insps_mat = np.vstack(
    all_trials["breath"]
)  # or use all_trials["insps_padded"] for insp only

for bin in range(0, len(edges) - 1):

    # get trials in bin
    lower, upper = edges[bin : bin + 2]
    ii_bin = (offsets_ms >= lower) & (offsets_ms < upper)
    assert sum(ii_bin) == hist[bin]
    breaths = insps_mat[ii_bin, :].T

    # plot
    fig, ax = plt.subplots()

    x = np.arange(*window) / fs * 1000

    ax.plot(
        x,
        breaths,
        linewidth=0.5,
        alpha=0.7,
    )

    ax.plot(
        x,
        breaths.mean(axis=1),
        linewidth=1,
        alpha=1,
        color="k",
    )

    ax.set(
        xlabel="time (ms, stim-aligned)",
        ylabel="amplitude",
        title=f"Offset: [{lower}, {upper})ms. Count: {sum(ii_bin)}",
    )

    if upper < 500:
        ax.set_xlim([-200, 500])

    fig.savefig(
        save_folder.joinpath(f"offset_distr-bin{bin}-{int(lower)}_{int(upper)}ms.jpg")
    )
    plt.close(fig)


# %%
# make umap embedding

insps_mat = np.vstack(all_trials["insps_padded"])

umap_name = "embedding3"
model = umap.UMAP(
    n_neighbors=10,
    min_dist=0.5,
    metric="correlation",
)

embedding = model.fit_transform(insps_mat)


# %%
# plot umap

all_insps = np.vstack(all_trials["ii_first_insp"]).T
offsets_ms = all_insps[1, :] / fs * 1000

fig, ax = plt.subplots()

sc = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    s=4,
    alpha=0.8,
    c=offsets_ms,
    cmap="viridis",
)

ax.set(
    xlabel="UMAP1",
    ylabel="UMAP2",
    title=umap_name,
)

cbar = fig.colorbar(sc, label="insp offset (ms, stim-aligned)")


# %%
# save umap plot & embedding

save_folder = pathlib.Path("./data/umap")

with open( save_folder.joinpath(f"{umap_name}.pickle"), "wb") as f:
    pickle.dump(
        {
            "model": model,
            "embedding": embedding,
        },
        f,
    )

fig.savefig(save_folder.joinpath(f"{umap_name}.jpg"))
