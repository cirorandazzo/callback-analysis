# %%
# all_trials-columns.py
#

import glob
from itertools import product
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import matplotlib.colors
import matplotlib.pyplot as plt

import umap

import hdbscan

# %%
# load all_trials data

fs = 44100
buffer_ms = 500

all_trials_path = Path(r".\data\umap\all_trials.pickle")


# ALL TRIALS
with open(all_trials_path, "rb") as f:
    all_trials = pickle.load(f)

# DEFINE WINDOW FOR CUTTING
fs = 44100

buffer_fr = int(buffer_ms * fs / 1000) + 1
all_insps = np.vstack(all_trials["ii_first_insp"]).T

window = (all_insps.min() - buffer_fr, all_insps.max() + buffer_fr)


all_trials

# %%
# VARIOUS UMAP INPUTS

plt.close("all")

breath_cols = [
    # "breath",
    "breath_norm",
    "insps_padded",
    "insps_unpadded",
    "insps_interpolated",
    # "breath_first_cycle",
    # "breath_first_cycle_unpadded",
]

trial = all_trials.iloc[0]

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13.3, 5.5))

# PLOT BREATH_NORM
ax = axs[0, 0]
plot_kwargs = dict(linewidth=1)

x = np.arange(*window) / fs * 1000


insp_on, insp_off = trial["ii_first_insp"] - window[0]
insp = trial["breath_norm"][insp_on:insp_off]

ax.plot(
    x,
    trial["breath_norm"],
    label="breath_norm",
    c="k",
    **plot_kwargs,
)
ax.plot(
    np.arange(*trial["ii_first_insp"]) / fs * 1000,
    insp,
    label="insp",
    c="r",
    **plot_kwargs,
)

ax.set(
    title="breath_norm",
    xlabel="time (ms, stim-aligned)",
    ylabel="normalized amplitude",
)


# insps_padded
ax = axs[0, 1]

x = np.arange(*window) / fs * 1000

ax.plot(
    x,
    trial["insps_padded"],
    c="k",
    **plot_kwargs,
)

insp = trial["insps_padded"][insp_on:insp_off]
ax.plot(
    np.arange(*trial["ii_first_insp"]) / fs * 1000,
    insp,
    label="insp",
    c="r",
    **plot_kwargs,
)

ax.set(
    title="insps_padded",
    xlabel="time (ms, stim-aligned)",
    ylabel="normalized amplitude",
)


# insps: end-padded
ax = axs[1, 0]

breath = trial["insps_padded_right_zero"]

x = np.arange(len(breath)) / fs * 1000

ax.plot(
    x,
    breath,
    c="k",
    **plot_kwargs,
)

insp = breath[ : np.ptp(trial["ii_first_insp"])]
x = np.arange(len(insp)) / fs * 1000

ax.plot(
    x,
    insp,
    c="r",
    **plot_kwargs
)

ax.set(
    title="insps_padded_right_zero",
    xlabel="time (ms, insp onset-aligned)",
    ylabel="normalized amplitude",
)


# insps_interpolated
ax = axs[1, 1]

insp = trial["insps_interpolated"]

x = np.linspace(0, 1, len(insp))

ax.plot(
    x,
    insp,
    c="r",
    **plot_kwargs,
)

ax.set(
    title="insps_interpolated",
    xlabel="time (normalized insp duration)",
    ylabel="normalized amplitude",
)

fig.tight_layout()


# %%
# UNUSED UMAP INPUT PANELS

# ===== INSPS_UNPADDED
# insp = trial["insps_unpadded"]

# x = np.arange(len(insp)) / fs * 1000

# ax.plot(
#     x,
#     insp,
#     c="r",
#     **plot_kwargs,
# )

# ax.set(
#     title="insps_unpadded",
#     xlabel="time (ms, insp onset-aligned)",
#     ylabel="normalized amplitude",
# )


# ===== breath_first_cycle_unpadded

# breath = trial["breath_first_cycle_unpadded"]
# x = np.arange(len(breath)) / fs * 1000

# ax.plot(
#     x,
#     breath,
#     c="k",
#     **plot_kwargs,
# )

# insp = breath[ : np.ptp(trial["ii_first_insp"])]
# x = np.arange(len(insp)) / fs * 1000

# ax.plot(
#     x,
#     insp,
#     c="r",
#     **plot_kwargs
# )

# ax.set(
#     title="breath_first_cycle_unpadded",
#     xlabel="time (ms, insp onset-aligned)",
#     ylabel="normalized amplitude",
# )

# %%

fig, ax = plt.subplots()

# ===== breath_first_cycle

breath = trial["breath_first_cycle"]
x = np.arange(*window) / fs * 1000

ax.plot(
    x,
    breath,
    c="k",
    **plot_kwargs,
)

ax.set(
    title="breath_first_cycle (padded)",
    xlabel="time (ms, stim-aligned)",
    ylabel="normalized amplitude",
)


# %%
# NORMALIZATION

plt.close('all')

plot_kwargs = dict(linewidth=0.5, alpha=0.7)

fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(13.3, 5.5))

x = np.arange(*window) / fs * 1000

# get first set of files
for fname, file_trials in all_trials.groupby("wav_filename"):
    break


file_trials["breath"].apply(
    lambda trial: axs[0].plot(x, trial, **plot_kwargs)
)

file_trials["breath_norm"].apply(
    lambda trial: axs[1].plot(x, trial, **plot_kwargs)
)


hlines_plot_kwargs = dict(
    color="k",
    linestyle="--",
    linewidth=0.75,
)

breath_min = np.min(file_trials["breath"].apply(np.min))

for ax, y_min in zip(axs, [breath_min, -1]):
    ax.axhline(y=0, **hlines_plot_kwargs)
    ax.axhline(y=y_min, **hlines_plot_kwargs)

axs[0].set(
    title="Centered breath trace",
    xlim=[-100, 500],
    xlabel="time (ms, stim-aligned)",
    ylabel="amplitude (centered)",
)

axs[1].set(
    title="Centered + normalized breath trace",
    xlabel="time (ms, stim-aligned)",
    ylabel="normalized amplitude",
)

fig.suptitle(f"{Path(fname).name}: all trials")
