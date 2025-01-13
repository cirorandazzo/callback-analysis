# %%
#

import glob
import os
import pickle

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt

from utils.audio import AudioObject
from utils.breath import segment_breaths
from utils.file import parse_birdname

# %%
# %load_ext autoreload
# %autoreload 1
# %matplotlib widget
# %aimport utils.audio
# %aimport utils.breath
# %aimport utils.callbacks

# %%
# get filelist
# taken from file `breaths-make_df-plot`

figure_save_folder = "./data/breath_figs/rolling_mean-multi"

default_bird = "rd56"  # only bird which fails name parsing
exist_ok = True  # False --> error out if folder already exists

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
# PLOT OPTIONS/FUNCTIONS
fs = 44100

plot_kwargs_exp = {
    "color": "r",
    "zorder": 3,
    "marker": "+",
}
plot_kwargs_insp = {
    "color": "b",
    "zorder": 3,
    "marker": "+",
}

plot_kwargs_distr_marker = {
    "marker": "+",
    "s": 16,
    "zorder": 3,
}

def plot_amplitude_dist(breath, ax, binwidth=100, leftmost=None, rightmost=None):
    # hist, edges = np.histogram(breath, bins=50, density=True)

    if leftmost is None:
        leftmost = min(breath) - 2 * binwidth

    if rightmost is None:
        rightmost = max(breath) + 2 * binwidth

    hist, edges = np.histogram(
        breath, bins=np.arange(leftmost, rightmost, binwidth), density=True
    )

    ax.stairs(hist, edges, fill=True)

    # 25 & 75th percentile: black lines
    ax.vlines(
        x=[np.percentile(breath, p) for p in (25, 75)],
        ymin=0,
        ymax=max(hist),
        color="k",
        linestyles="--",
        alpha=0.5,
        zorder=3,
        label="p25 & p75",
    )

    median_multiples = (1, 1.5, 2)
    # median & multiples: red lines
    ax.vlines(
        x=[q * np.median(breath) for q in median_multiples],
        ymin=0,
        ymax=max(hist),
        color="r",
        linestyles=":",
        alpha=0.5,
        zorder=3,
        label=f"median * {median_multiples}",
    )


# %%
# threshold from amplitude distr fit - rolling min subtraction

# ideally at breathing rate ,so you always normalize to most recent insp
window_length = int(0.5 * fs)


# lowpass
b_lp, a_lp = butter(N=2, Wn=50, btype="low", fs=fs)


def make_rolling_mean_plot(
    x,
    breath_lowpass,
    breath_roll_min_subtr,
    rolling_min,
    x_dist,
    dist_kde,
    top2,
    trough,
    exps,
    insps,
    binwidth = 50,
    **unused_kwargs,
):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # distributions (first column)
    ax_lp_dist = axs[0, 0]  # for lowpass amplitude distribution
    ax_rm_dist = axs[1, 0]  # for rolling mean subtr'd amplitude distribution

    # breath trace (second column)
    ax_lp = axs[0, 1]  # for lowpass waveform
    ax_rm = axs[1, 1]  # for rolling mean subtr'd waveform

    # plot distributions
    plot_amplitude_dist(breath_lowpass, ax_lp_dist, binwidth=binwidth)

    plot_amplitude_dist(breath_roll_min_subtr, ax_rm_dist, binwidth=binwidth)

    ax_rm_dist.plot(x_dist, dist_kde, color="k")  
    ax_rm_dist.scatter(  # mark highest 2 peaks
        x_dist[top2],
        dist_kde[top2],
        color="#EE893B",
        label="peaks",
        **plot_kwargs_distr_marker,
    )
    ax_rm_dist.scatter(  # mark trough between those peaks
        x_dist[trough],
        dist_kde[trough],
        color="r",
        label="threshold",
        **plot_kwargs_distr_marker,
    )

    # plot lowpass waveform
    ax_lp.plot(x, breath_lowpass, linewidth=0.5, label="lowpass breath")
    ax_lp.plot(x, rolling_min, linewidth=0.5, color="#EF6F6C", label="rolling min")
    ax_lp.scatter(
        x[exps],
        breath_lowpass[exps],
        label="exp",
        **plot_kwargs_exp,
    )
    ax_lp.scatter(
        x[insps],
        breath_lowpass[insps],
        label="insp",
        **plot_kwargs_insp,
    )

    # plot rolling min subtracted waveform
    ax_rm.plot(x, breath_roll_min_subtr, linewidth=0.5, label="rolling min subtracted")
    ax_rm.scatter(
        x[exps],
        breath_roll_min_subtr[exps],
        **plot_kwargs_exp,
    )
    ax_rm.scatter(
        x[insps],
        breath_roll_min_subtr[insps],
        **plot_kwargs_insp,
    )

    return fig, axs

processed_data = []

for i_file, file in enumerate(files):
    # ===== LOAD & FILTER ===== #
    breath = AudioObject.from_wav(file, channels=1, b=b_lp, a=a_lp)
    x = breath.get_x()
    breath_lowpass = breath.audio_filt  # get lowpass filt

    # ===== ROLLING MIN ===== #
    rolling_min = np.array(
        pd.Series(breath_lowpass).rolling(window=window_length).min()
    )
    # backfill first window with first non-nan value
    i_first_non_nan = int(np.flatnonzero(~np.isnan(rolling_min))[0])
    rolling_min[:i_first_non_nan] = rolling_min[i_first_non_nan]
    breath_roll_min_subtr = breath_lowpass - rolling_min

    # ===== SPLINE FIT ===== #
    # plot fitted distribution
    x_dist = np.linspace(breath_roll_min_subtr.min(), breath_roll_min_subtr.max(), 100)
    kde = gaussian_kde(breath_roll_min_subtr)
    dist_kde = kde(x_dist)

    x_peaks = find_peaks(dist_kde)[0]

    # push closest peak to 0; should be the case with rolling min subtraction
    if dist_kde[0] > dist_kde[min(x_peaks)]:
        x_peaks[x_peaks.argmin()] = 0

    top2 = sorted(
        x_peaks[np.argsort(dist_kde[x_peaks])][-2:]
    )  # get indices of highest 2 peaks.

    trough = top2[0] + np.argmin(
        dist_kde[np.arange(*top2)]
    )  # location of minimum value between these points

    # ===== SEGMENT BREATHS ===== #
    exps, insps = segment_breaths(
        breath_roll_min_subtr,
        do_filter=False,
        threshold=lambda x: x_dist[trough],
        fs=None,
    )

    # ===== STORE DATA ===== #

    data_dict = dict(
        i_file=i_file,
        file=file,
        x=x,
        breath_lowpass=breath_lowpass,
        breath_roll_min_subtr=breath_roll_min_subtr,
        rolling_min=rolling_min,
        x_dist=x_dist,
        dist_kde=dist_kde,
        top2=top2,
        trough=trough,
        exps=exps,
        insps=insps,
    )

    processed_data.append(data_dict)

    # ===== MAKE BIRD FOLDER ===== #
    root = os.path.splitext(file)[0]  # entire filename without extension
    basename = os.path.split(root)[-1]

    try:
        birdname = parse_birdname(root)
        # bird = "rd56"
    except TypeError:
        if default_bird is not None:
            birdname = default_bird
        else:
            raise TypeError(
                f"Couldn't parse birdname from: {root}\nIf this bird just has one band, you should find this and hard-code its name."
            )

    bird_folder = os.path.join(figure_save_folder, birdname)
    os.makedirs(bird_folder, exist_ok=True)

    # ===== PLOTS ===== #
    fig_savepath = os.path.join(bird_folder, f"{basename}.jpg")

    fig_multiplot, axs_multiplot = make_rolling_mean_plot(**data_dict)
    axs_multiplot.ravel()[0].set(title=basename)

    fig_multiplot.savefig(fig_savepath)


# %%

pickle_file = os.path.join( figure_save_folder, "rolling_mean_subtracted.pickle")

df = pd.DataFrame.from_records(processed_data).set_index("i_file")

with open(pickle_file, "wb") as f:
    pickle.dump(df, f)

print(f"Successfully dumped data to: {pickle_file}")

# %%

df
