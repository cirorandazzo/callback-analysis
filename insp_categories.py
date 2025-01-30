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
from itertools import product
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
# get first insp + following exp for all trials
#
# stim-aligned & in samples

max_insp_length_s = 0.5

all_trials["ii_first_insp"] = all_trials.apply(
    get_first_breath_segment,
    axis=1,
    breath_type="insp",
    fs=fs,
    buffer_s=0,
    return_stim_aligned=True,
    return_unit="samples",
)

# reject trials without insp
#
# these occur at end of callbacks - final trial cuts off immediately post-stim

ii_no_insps = all_trials["ii_first_insp"].isna()

rejected = all_trials.loc[ii_no_insps]
all_trials = all_trials.loc[~ii_no_insps]


# reject trials with insp too long
ii_good_duration = all_trials["ii_first_insp"].apply(lambda x: np.ptp(x) <= max_insp_length_s * fs)

rejected = pd.concat([rejected, all_trials.loc[~ii_good_duration]])
all_trials = all_trials.loc[ii_good_duration]


# get exp following inspiration
def get_next_exp(trial, fs):

    insp_offset_s = trial["ii_first_insp"][1] / fs

    next_exp = get_first_breath_segment(
        trial,
        breath_type="exp",
        fs=fs,
        earliest_allowed_onset=insp_offset_s,
        buffer_s=0,
        return_stim_aligned=True,
        return_unit="samples",
    )

    return next_exp


all_trials["ii_next_exp"] = all_trials.apply(get_next_exp, axis=1, fs=fs)

all_trials

# %%
# report rejected trials

print("Rejected trials: no inspiration, or too long.")
rejected

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


def get_breath_for_trial(trial, breath, window, fs, centered=False):

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

    if centered:
        cut_breath = cut_breath - trial["breath_zero_point"]

    return cut_breath


for wav_filename, file_trials in all_trials.groupby("wav_filename"):

    breath = AudioObject.from_wav(wav_filename, channels=1, b=b, a=a).audio_filt

    all_trials.loc[file_trials.index, "breath"] = file_trials.apply(
        get_breath_for_trial,
        axis=1,
        breath=breath,
        window=window,
        fs=fs,
        centered=True,  # NOTE: this was initially false.
    )
    # TODO: normalize between file zero point & min?

all_trials


# %%
# get insp only trace (zero-padded to window size)


def get_insp_trace(trial, window, pad_value=0, pad=True):

    breath = trial["breath"].copy()
    insp_on, insp_off = trial["ii_first_insp"]

    # CHECKS
    pre_insp = insp_on - window[0]
    assert pre_insp > 0, f"{window[0]} | {insp_on}"

    post_insp = insp_off - window[1]
    assert post_insp < 0, f"{window[1]} | {insp_off}"

    # DO PADDING
    if pad:
        breath[:pre_insp] = pad_value
        breath[post_insp:] = pad_value
    else:
        breath = breath[pre_insp:post_insp]


    return breath


all_trials.loc[:, "insps_padded"] = all_trials.apply(
    get_insp_trace,
    axis=1,
    window=window,
)

all_trials.loc[:, "insps_unpadded"] = all_trials.apply(
    get_insp_trace,
    axis=1,
    window=window,
    pad=False,
)

all_trials


# %%
# putative calls

exp_window_ms = 300  # window after insp offset
threshold = 2  # times magnitude of inspiration

exp_window_fr = int(exp_window_ms * fs / 1000)


def check_call(trial, window, exp_window_fr, threshold):
    # indices of insp in trial["breath"]
    insp_on, insp_off = trial["ii_first_insp"] - window[0]

    post_insp_window = trial["breath"][insp_off : insp_off + exp_window_fr]

    putative_call = any(post_insp_window > threshold * abs(min(trial["insps_unpadded"])))

    return putative_call


# sample plot to show segments
def plot_segments(trial, window, exp_window_fr, threshold):
    fig, ax = plt.subplots()

    insp_on, insp_off = trial["ii_first_insp"] - window[0]

    ax.plot(
        trial["breath"],
        label="breath",
    )
    ax.plot(
        np.arange(insp_on, insp_off),
        trial["insps_unpadded"],
        label="insp",
    )

    ii_exp = np.arange(insp_off, insp_off + exp_window_fr)
    ax.plot(
        ii_exp,
        trial["breath"][ii_exp],
        label="exp window",
    )

    ax.hlines(
        xmin=0,
        xmax=np.ptp(window),
        y=threshold * abs(min(trial["insps_unpadded"])),
        colors="k",
        linestyles="--",
        linewidths=0.5,
        label="threshold",
    )

    ax.legend()

    return fig, ax

# putative call based on amplitude
#
# previously: threshold based on 98th percentile of breath waveform per file
# here: threshold of (2 * abs(insp_amp)) in 300ms post insp. no windowing
#
# these correspond well, apparently. however, this seems more robust. will use this going forward

# to keep previous putative_call
# all_trials.rename(columns={"putative_call": "putative_call-pct"}, inplace=True)

all_trials["putative_call"] = all_trials.apply(
    check_call,
    axis=1,
    window=window,
    exp_window_fr=exp_window_fr,
    threshold=threshold,
)


# %%
# end-pad calls with discrete call label

def pad_insps(trial, pad_to, pad_value):

    insp = trial["insps_unpadded"]
    pad_length = pad_to - len(insp)

    padded = np.pad(insp, [0, pad_length], mode="constant", constant_values=pad_value)

    return padded

all_trials["insps_padded_call_discrete"] = all_trials.apply(
    lambda trial: pad_insps(
        trial,
        pad_to=max(all_trials["insps_unpadded"].apply(len)),  # max insp length
        pad_value=trial["putative_call"] * 1000,
    ),
    axis=1,
)

all_trials["insps_padded_right_zero"] = all_trials.apply(
    lambda trial: pad_insps(
        trial,
        pad_to=max(all_trials["insps_unpadded"].apply(len)),  # max insp length
        pad_value=0,
    ),
    axis=1,
)

# %%
# plot all aligned traces

fig, ax = plt.subplots()
for trace in all_trials["breath"]:
    ax.plot(np.arange(*window), trace)

ax.set(
    title="breaths",
    xlabel="samples (stim-aligned)",
    ylabel="amplitude",
)

plt.show()

# %%
# plot all aligned traces - insp only

fig, ax = plt.subplots()
for trace in all_trials["insps_padded"]:
    ax.plot(np.arange(*window), trace)

ax.set(
    title="padded insps",
    xlabel="samples (stim-aligned)",
    ylabel="amplitude",
)

plt.show()


# %%
# plot all call-discretized traces: call vs no call trials

dfs = {
    "no call": all_trials.loc[~all_trials["putative_call"]],
    "call": all_trials.loc[all_trials["putative_call"]],
}

fig, axs = plt.subplots(ncols=len(dfs.keys()), sharey=True)

for (key, df), ax in zip(dfs.items(), axs):
    for trace in df["insps_padded_call_discrete"]:
        ax.plot(trace)
    ax.set(title=key)

fig.suptitle("Discretized call")
fig.tight_layout()

# %%
# plot duration histograms

hist_kwarg = dict(alpha=0.5, color="green")

def plot_duration_hist(all_trials, subtitle, binwidth=10, hist_min=0,hist_max=860, **hist_kwarg):
    all_insps = np.vstack(all_trials["ii_first_insp"]).T

    durations_ms = (all_insps[1, :] - all_insps[0, :]) / fs * 1000

    fig, ax = plt.subplots()

    # ax.hist(durations_ms, **hist_kwarg)

    hist, edges = np.histogram(
        durations_ms,
        bins=np.arange(hist_min, hist_max, binwidth),
    )
    ax.stairs(hist, edges, fill=True,**hist_kwarg)

    ax.set(
        title=f"first inspiration duration: {subtitle}",
        xlabel="duration (ms)",
        ylabel="count",
        xlim=[-10,360],
    )

    return fig, ax


# all birds merged
plot_duration_hist(all_trials, "all birds", **hist_kwarg)

# by bird
for birdname, all_trials_bird in all_trials.groupby(level="birdname"):
    plot_duration_hist(all_trials_bird, birdname, **hist_kwarg)


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
# plot breath traces by insp bin

save_folder = pathlib.Path("./data/insp_bins-offset")

histogram_kwarg = dict(bins=40, range=(0, 840))


def plot_hist_and_binned(all_trials, label, fs, window, save_folder, color_by=None, **histogram_kwarg):

    all_insps = np.vstack(all_trials["ii_first_insp"]).T
    offsets_ms = all_insps[1, :] / fs * 1000

    # make & plot histogram
    hist, edges = np.histogram(offsets_ms, **histogram_kwarg)

    fig, ax = plt.subplots()

    ax.stairs(hist, edges)
    ax.set(
        xlabel="Insp offset (ms)",
        ylabel="Count",
        title=f"Insp offset distribution: {label}",
    )

    fig.savefig(save_folder.joinpath("offset_distr.jpg"))
    plt.close(fig)

    # plot breath traces by bin
    insps_mat = np.vstack(
        all_trials["breath"]
    )  # or use all_trials["insps_padded"] for insp only

    if color_by is not None:
        categories = sorted(all_trials.index.get_level_values(color_by).unique())

    for bin in range(0, len(edges) - 1):
        # get trials in bin
        lower, upper = edges[bin : bin + 2]
        ii_bin = (offsets_ms >= lower) & (offsets_ms < upper)
        assert sum(ii_bin) == hist[bin]
        breaths = insps_mat[ii_bin, :].T

        # plot
        fig, ax = plt.subplots()

        x = np.arange(*window) / fs * 1000

        trace_kwargs = dict(
            linewidth=0.5,
            alpha=0.7,
        )

        # plot all traces in this bin
        if color_by is not None:
            for i, category in enumerate(categories):
                ii_cat = all_trials.iloc[ii_bin].index.get_level_values(color_by) == category

                ax.plot(
                    x,
                    breaths[:, ii_cat],
                    c=f"C{i}",
                    **trace_kwargs
                )
        else:
            ax.plot(
                x,
                breaths,
                **trace_kwargs
            )

        # plot mean trace
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
            save_folder.joinpath(
                f"{label}-offset_distr-bin{bin}-{int(lower)}_{int(upper)}ms.jpg"
            )
        )
        plt.close(fig)


all_save_folder = save_folder.joinpath("all_birds")
os.makedirs(all_save_folder)

# for all trials
a = plot_hist_and_binned(
    all_trials, "all_birds", fs, window, all_save_folder, color_by="birdname",**histogram_kwarg
)

# for individual birds
for birdname, all_trials_bird in all_trials.groupby(level="birdname"):

    bird_save_folder = save_folder.joinpath(birdname)
    os.makedirs(bird_save_folder)

    plot_hist_and_binned(
        all_trials, birdname, fs, window, bird_save_folder, **histogram_kwarg
    )


# %%
# make umap embedding

save_folder = pathlib.Path("./data/umap")

umap_params = dict(
    insp_col_name=[
        "insps_padded",
        "insps_padded_call_discrete",
        "insps_padded_right_zero",
    ],
    n_neighbors=[3, 5, 10, 20],
    min_dist=[0.01, 0.1, 0.5, 1],
    metrics=[
        "cosine",
        "correlation",
        "euclidean",
    ],
)

# make parameter combinations
conditions = []
for condition in product(*umap_params.values()):
    conditions.append({k: v for k, v in zip(umap_params.keys(), condition)})


for i, condition in enumerate(conditions):

    umap_name = f"embedding{i}"

    with open(save_folder.joinpath(f"log.txt"), "a") as f:
        f.write(f"embedding{i}")
        f.write(str(condition))


    insp_type = condition.pop("insp_col_name")
    insps_mat = np.vstack(all_trials[insp_type])

    model = umap.UMAP(
        n_neighbors=10,
        min_dist=0.5,
        metric="correlation",
    )

    embedding = model.fit_transform(insps_mat)


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


    # save umap plot & embedding

    save_folder = pathlib.Path("./data/umap")

    with open(save_folder.joinpath(f"{umap_name}.pickle"), "wb") as f:
        pickle.dump(
            {
                "model": model,
                "embedding": embedding,
            },
            f,
        )

    fig.savefig(save_folder.joinpath(f"{umap_name}.jpg"))
    plt.close(fig)
