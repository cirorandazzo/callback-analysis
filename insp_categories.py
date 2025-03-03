# %%
#
# insp categories
#
# idea: what's the basis set of inspiratory patterns?
# (1) look at distribution of where first insp occurs (should be random)
# (2) categories
#   (a) first pass: binned averages
#   (b) second pass: umap
#
# Note: everything after cell "end-pad calls" and before cell "make umap embeddings" makes plots - you can skip these if you just want new embeddings.

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

breath_field = "breath_norm"  # will be used for plots and such.

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

all_trials["ii_first_insp_window_aligned"] = all_trials["ii_first_insp"].apply(lambda x: x-window[0])

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

    zero_point = file_trials["breath_zero_point"].unique()
    assert len(zero_point) == 1
    zero_point = zero_point[0]

    breath = AudioObject.from_wav(wav_filename, channels=1, b=b, a=a).audio_filt
    breath -= zero_point

    all_trials.loc[file_trials.index, "breath"] = file_trials.apply(
        get_breath_for_trial,
        axis=1,
        breath=breath,
        window=window,
        fs=fs,
        centered=False
    )
    
    # breath_norm: map [biggest_insp, zero_point] --> [-1, 0]
    insp_magnitude = np.abs(breath.min())

    all_trials.loc[file_trials.index, "breath_norm"] = all_trials.loc[file_trials.index, "breath"].apply(
        lambda x: x / insp_magnitude
    )

all_trials


# %%
# get insp only trace (zero-padded to window size)

def get_trace(trial, window, pad_value=0, pad=True, breath_field="breath", which="first_insp"):

    breath = trial[breath_field].copy()

    if which == "first_insp":
        on, off = trial["ii_first_insp"]
    elif which == "next_exp":
        on, off = trial["ii_next_exp"]
    elif which == "first_cycle":
        on = trial["ii_first_insp"][0]
        off = trial["ii_next_exp"][1]
    else:
        raise KeyError(f"Unknown mode: {which}")

    # CHECKS
    i_pre = on - window[0]
    assert i_pre > 0, f"{window[0]} | {on}"

    i_post = off - window[1]
    assert i_post < 0, f"{window[1]} | {off}"

    # DO PADDING
    if pad:
        breath[:i_pre] = pad_value
        breath[i_post:] = pad_value
    else:
        breath = breath[i_pre:i_post]


    return breath


all_trials.loc[:, "insps_padded"] = all_trials.apply(
    get_trace,
    axis=1,
    window=window,
    breath_field=breath_field,
)

all_trials.loc[:, "insps_unpadded"] = all_trials.apply(
    get_trace,
    axis=1,
    window=window,
    pad=False,
    breath_field=breath_field,
)

# first insp and following exp, padded to window.
all_trials["breath_first_cycle"] = all_trials.apply(
    get_trace,
    axis=1,
    which="first_cycle",
    pad=True,
    window=window,
    breath_field=breath_field,
)


# interpolate to stretch all insps to same length
max_length = max(all_trials["insps_unpadded"].apply(len))

all_trials.loc[:, "insps_interpolated"] = all_trials["insps_unpadded"].apply(
    lambda trial: np.interp(
        np.linspace(0, len(trial), max_length),
        np.arange(len(trial)),
        trial,
    )
)

all_trials


# %%
# redefine putative calls

threshold = 1.1  # absolute. most useful for normalized trace


def check_call(trial, window, threshold, return_magnitudes=False, breath_field="breath_norm"):
    # indices of exp in trial["breath"]
    exp_on, exp_off = trial["ii_next_exp"] - window[0]
    exp_window = trial[breath_field][exp_on : exp_off]

    if return_magnitudes: 
        insp_on, insp_off = trial["ii_first_insp"] - window[0]
        insp_window = trial[breath_field][insp_on : insp_off]

        return (np.abs(insp_window.min()), exp_window.max())

    else:
        putative_call = exp_window.max() >= threshold

        return putative_call


# sample plot to show segments
def plot_segments(
    trial,
    window,
    exp_window_fr,
    threshold,
    breath_field="breath",
    ax=None,
    legend=True,
    **plot_kwargs,
):

    import warnings

    warnings.warn("This no longer shows the same algorithm that check_call uses!")

    if ax is None:
        fig, ax = plt.subplots()

    insp_on, insp_off = trial["ii_first_insp"] - window[0]

    insp = trial[breath_field][insp_on : insp_off]

    ax.plot(
        trial[breath_field],
        label=breath_field,
        c="k",
        **plot_kwargs,
    )
    ax.plot(
        np.arange(insp_on, insp_off),
        insp,
        label="insp",
        c="b",
        **plot_kwargs,
    )

    ii_exp = np.arange(insp_off, insp_off + exp_window_fr)
    ax.plot(
        ii_exp,
        trial[breath_field][ii_exp],
        label="exp window",
        c="g",
        **plot_kwargs,
    )

    if threshold is not None:
        ax.hlines(
            xmin=0,
            xmax=np.ptp(window),
            y=threshold * abs(insp.min()),
            colors="k",
            linestyles="--",
            linewidths=0.5,
            label="threshold",
        )

    if legend:
        ax.legend()

    return ax


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
    threshold=threshold,
    breath_field="breath_norm",
)

# %%
# end-pad calls

def pad_insps(trial, pad_to, pad_value):

    insp = trial["insps_unpadded"]
    pad_length = pad_to - len(insp)

    padded = np.pad(insp, [0, pad_length], mode="constant", constant_values=pad_value)

    return padded

all_trials["insps_padded_call_discrete"] = all_trials.apply(
    lambda trial: pad_insps(
        trial,
        pad_to=max(all_trials["insps_unpadded"].apply(len)),  # max insp length
        pad_value=trial["putative_call"],
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
# magnitude distr for putative call

magnitudes = all_trials.apply(
    check_call,
    axis=1,
    window=window,
    threshold=threshold,
    return_magnitudes=True,
    breath_field="breath_norm",
)

magnitudes = pd.DataFrame(
    [a for a in magnitudes.values],
    columns=["mag_insp", "mag_exp"],
    index=all_trials.index,
)

fig, ax = plt.subplots()

h, xedge, yedge, im = ax.hist2d(magnitudes["mag_insp"], magnitudes["mag_exp"], bins=50, cmap="cividis")
fig.colorbar(im, ax=ax, label="count")

ax.set(
    xlabel="insp magn",
    ylabel=f"magn, next exp",
)

insp_max = magnitudes["mag_insp"].max()

# insp scaled thresholds
for t in [2, 3, 4]:
    ax.plot([0, insp_max], [0, t * insp_max], label=f"threshold ({t}x)")

# const threshold
y = 1.1
ax.axhline(y=y, label=f"threshold (const={y})", c="r", linewidth=0.5, linestyle="--")

ax.legend(loc="upper right")

plt.show()

# %%
# look at traces for a range of exp magnitudes
# 
# amplitude in exp_window_ms after first inspiration

exp_window_fr = (300 / 1000) * fs
exp_amps_to_plot = [1,2]
ii_exp_range = (magnitudes["mag_exp"] >= exp_amps_to_plot[0]) & (
    magnitudes["mag_exp"] <= exp_amps_to_plot[1]
)

fig, ax = plt.subplots()
all_trials.loc[ii_exp_range].apply(
    plot_segments,
    axis=1,
    window=window,
    exp_window_fr=exp_window_fr,
    breath_field="breath_norm",
    ax=ax,
    # suppress these bits
    threshold=None,
    legend=False,
    linewidth=0.5,
)

ax.axvline(x=abs(window[0]), c="r", linestyle="--", label="stim")

ax.set(
    title=f"putative_calls bins: exp amplitude {exp_amps_to_plot} (n={sum(ii_exp_range)})",
    xlabel="samples",
    ylabel="amplitude",
)

plt.show()

# %%
# plot all aligned traces

fig, ax = plt.subplots()
for trace in all_trials[breath_field]:
    ax.plot(np.arange(*window), trace)

ax.set(
    title=breath_field,
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
# plot 2d hist of onset/offset

bin_width_f = 0.02 * fs

fig, ax = plt.subplots()

edges = (  # uses same bins for onset & offset to ensure axis 
    np.arange(
        all_insps.ravel().min(), all_insps.ravel().max() + bin_width_f, bin_width_f
    )
    / fs
    * 1000
)

hist, x_edges, y_edges, im = ax.hist2d(
    x=all_insps[0, :] / fs * 1000,  # onsets_ms
    y=all_insps[1, :] / fs * 1000,  # offsets_ms
    cmap="hot",
    bins=edges,
)

ax.axis("equal")
plt.box(False)

ax.set(
    xlabel="onsets (ms, stim-aligned)",
    ylabel="offsets (ms, stim-aligned)",
    title="First inspiration timing",
)

fig.colorbar(im, ax=ax, label="Number of breaths")


# %%
# plot breath traces by insp bin

save_folder = pathlib.Path("./data/insp_bins-offset")

histogram_kwarg = dict(bins=40, range=(0, 840))


def plot_hist_and_binned(all_trials, label, fs, window, save_folder, breath_field="breath", color_by=None, **histogram_kwarg):

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
        all_trials[breath_field]
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
    all_trials, "all_birds", fs, window, all_save_folder, color_by="birdname", breath_field=breath_field, **histogram_kwarg
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
        "insps_interpolated",
        "insps_padded_call_discrete",
        "breath_first_cycle",
    ],
    n_neighbors=[2, 10, 20, 70],
    min_dist=[0, 1, 10],
    metric=[
        "cosine",
        "correlation",
        "euclidean",
    ],
)

# make parameter combinations
conditions = []
for condition in product(*umap_params.values()):
    conditions.append({k: v for k, v in zip(umap_params.keys(), condition)})

errors = {}

# save all_trials
with open(save_folder.joinpath(f"all_trials.pickle"), "wb") as f:
    pickle.dump(all_trials, f)

# run gridsearch
for i, condition in enumerate(conditions):

    umap_name = f"embedding{i}"

    if os.path.exists(save_folder.joinpath(f"{umap_name}.pickle")):
        print(f"#{i} exists! Skipping...")
        continue

    print(f"Embedding {i} / {len(conditions)}: {condition}")

    with open(save_folder.joinpath(f"log.txt"), "a") as f:
        f.write(f"- embedding{i}:\n")

        for k,v in condition.items():
            f.write(f"  - {k}: {v}\n")

    insp_type = condition.pop("insp_col_name")
    insps_mat = np.vstack(all_trials[insp_type])

    try:
        model = umap.UMAP(**condition)
        embedding = model.fit_transform(insps_mat)
    except Exception as e:
        errors[i] = e
        print(f"Error on #{i}! Skipping...")
        continue

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
