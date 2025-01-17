# %%
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 1
# %matplotlib widget
# %aimport utils.audio
# %aimport utils.breath
# %aimport utils.callbacks

from utils.audio import AudioObject
from utils.breath import segment_breaths, make_notmat_vars, plot_breath_callback_trial, plot_amplitude_dist
from utils.callbacks import call_mat_stim_trial_loader
from utils.evfuncs import segment_notes
from utils.file import parse_birdname, parse_parameter_from_string
from utils.video import get_triggers_from_audio

# %%
# get filelist

paths = [
    r"M:/eszter/behavior/air sac calls/HVC lesion/aspiration/rd99rd72/preLesion/callbacks/rand/230215/*-B*.wav",
    r"M:/eszter/behavior/air sac calls/HVC lesion/aspiration/pk19br8/preLesion/callback/rand/**/*-B*.wav",
    r"M:/eszter/behavior/air sac calls/HVC lesion/aspiration/rd56/preLesion/callbacks/male_230117/*-B*.wav",
    r"M:/eszter/behavior/air sac calls/HVC lesion/aspiration/rd57rd97/preLesion/callbacks/male_230117/*-B*.wav",
]

# `*-B*` excludes "-PostBlock"

# get all files matching above paths
files = [file for path in paths for file in sorted(glob.glob(os.path.normpath(path)))]

assert len(files) != 0, "No files found!"

print("Files: ")
for i, f in enumerate(files):
    print(f"{i}. {os.path.split(f)[1]}")


# %%
# plot distributions of filtered waveform

fs = 44100
b, a = butter(N=2, Wn=50, btype="low", fs=fs)

plot_kwargs_distr_marker = {
    "marker": "+",
    "s": 100,
    "linewidth": 2,
    "zorder": 3,
}

figure_save_folder = "./data/distributions"

for f in files:

    basename = os.path.splitext(os.path.basename(f))[0]

    try:
        birdname = parse_birdname(basename)
    except TypeError:
        birdname = "default"

    bird_folder = os.path.join(figure_save_folder, birdname)
    os.makedirs(bird_folder, exist_ok=True)

    # load audio
    channels = AudioObject.from_wav(
        f, channels="all", channel_names=["audio", "breathing", "trigger"]
    )

    assert fs == channels[1].fs, "Wrong sample rate!"

    channels[1].filtfilt(b, a)  # filter breathing
    breath = channels[1].audio_filt

    fig, ax = plt.subplots()
    plot_amplitude_dist(breath, ax, median_multiples=None, percentiles=None)

    x_dist = np.linspace(breath.min(), breath.max(), 100)
    kde = gaussian_kde(breath)
    dist_kde = kde(x_dist)

    x_peaks = find_peaks(dist_kde)[0]

    top2 = sorted(
        x_peaks[np.argsort(dist_kde[x_peaks])][-2:]
    )  # get indices of highest 2 peaks.

    trough = top2[0] + np.argmin(
        dist_kde[np.arange(*top2)]
    )  # location of minimum value between these points

    points = top2 + [trough]

    ax.plot(x_dist, dist_kde, color="k")

    ax.scatter(  # mark trough between those peaks
        x_dist[points],
        dist_kde[points],
        color="r",
        label="peaks & threshold",
        **plot_kwargs_distr_marker,
    )

    ax.vlines(
        x_dist[points],
        ymin=0,
        ymax=dist_kde[points],
        color="r",
        linewidth=1,
        linestyle="--",
    )

    ax.set(
        title=basename,
        xlabel="breath amplitude",
        ylabel="density",
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig( os.path.join(bird_folder, f"{basename}.jpg"), dpi=300)

    plt.close(fig)
