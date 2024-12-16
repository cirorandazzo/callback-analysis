# %%
#

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, sosfiltfilt, savgol_filter
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 1
# %matplotlib widget
# %aimport utils.audio
# %aimport utils.breath
# %aimport utils.callbacks

from utils.audio import AudioObject, plot_spectrogram
from utils.breath import segment_breaths, make_notmat_vars, plot_breath_callback_trial
from utils.callbacks import call_mat_stim_trial_loader
from utils.evfuncs import segment_notes
from utils.file import parse_birdname, parse_parameter_from_string
from utils.video import get_triggers_from_audio

# %%
# files

files = [
    # loud file: song
    r"M:\eszter\behavior\air sac calls\HVC lesion\aspiration\rd57rd97\preLesion\callbacks\male_230117\rd57rd97202311710026-Block0.wav",
    # medium file: many calls
    r"M:\eszter\behavior\air sac calls\HVC lesion\aspiration\pk19br8\preLesion\callback\rand\male_230217\pk19br820232177107-Block1.wav",
    # quiet file: nothing
    r"M:\eszter\behavior\air sac calls\HVC lesion\aspiration\rd56\preLesion\callbacks\male_230117\rd5720231178261-Block19.wav",
]

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


print(f"{len(files)} files.")

# %%
# example file structure

file = files[2]

channels = AudioObject.from_wav(
    file, channels="all", channel_names=["audio", "breath", "trigger"]
)

print("50Hz lowpass")
b, a = butter(N=2, Wn=5, btype="low", fs=fs)
# print("0.2 - 50Hz bp")
# b, a = butter(N=2, Wn=[0.2,50], btype="bandpass", fs=fs)

for c in channels:
    if c.name == "audio":
        c.filtfilt_butter_default()
    elif c.name == "breath":
        assert c.fs == fs
        c.filtfilt(b, a)

fig, axs = plt.subplots(nrows=len(channels), sharex=True)

for c, ax in zip(channels, np.ravel(axs)):
    if c.name == "audio":
        c.plot_spectrogram(ax=ax, cmap="magma")

        ax.set(ylim=(500, 15000))
    else:
        ax.plot(c.get_x(), c.audio)

axs[-1].set(
    xlabel="time (s)",
)

plt.show()
# %%
# plot waveforms

fig, axs = plt.subplots(nrows=len(files), sharex=True)

for file, ax in zip(files, axs.ravel()):
    breath = AudioObject.from_wav(file, channels=1, b=b, a=a)

    # refilter (useful if testing out filter options)
    breath.filtfilt(b, a)

    # ax.plot(breath.get_x(), breath.audio_filt, linewidth=0.5)
    ax.plot(breath.get_x(), breath.audio, linewidth=0.5)
    ax.set(title=os.path.basename(file))

    # median
    ax.plot(
        [0, breath.get_length_s()],
        np.median(breath.audio_filt) * np.array([1, 1]),
        linestyle=":",
        color="r",
    )

axs[int(len(files) / 2)].set(ylabel="amplitude")
axs[-1].set(xlabel="time since block onset (s)")
fig.tight_layout()

# %%
# plot rolling stdev
window_size = 1 * fs

fig, axs = plt.subplots(nrows=len(files), sharex=True)

for file, ax in zip(files, axs.ravel()):
    ax.set(title=os.path.basename(file))

    breath = AudioObject.from_wav(file, channels=1, b=b, a=a)
    stdev = pd.Series(breath.audio_filt).rolling(window=window_size).std()

    # PLOT STDEV
    # ax.plot(
    #     breath.get_x(),
    #     stdev,
    #     color=(0.1, 0.2, 0.1, 1),
    #     lw=0.5,
    # )
    # axs[ int(len(files)/2)].set(ylabel="rolling stdev (1s window)")

    # PLOT BREATH
    ax.plot(breath.get_x(), breath.audio_filt, linewidth=0.5)
    axs[int(len(files) / 2)].set(ylabel="amplitude")

    # PLOT WINDOW
    i_min_stdev = stdev.argmin()
    ax.plot(
        breath.get_x()[[i_min_stdev, i_min_stdev - window_size]],
        stdev[i_min_stdev] * np.array([1, 1]),
        marker="+",
        zorder=3,
        color="red",
    )

ax.set(xlabel="time since block onset (s)")

fig.tight_layout()


# %%
# plot amplitude histograms


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
    )

    # median & multiples: red lines
    ax.vlines(
        x=[q * np.median(breath) for q in (1, 1.5, 2)],
        ymin=0,
        ymax=max(hist),
        color="r",
        linestyles=":",
        alpha=0.5,
        zorder=3,
    )


fig, axs = plt.subplots(nrows=len(files), sharex=True)

for file, ax in zip(files, axs.ravel()):
    breath = AudioObject.from_wav(file, channels=1, b=b, a=a)
    plot_amplitude_dist(breath.audio_filt, ax)
    ax.set(title=os.path.basename(file), ylabel="density")

ax.set(xlabel="breath amplitude")

fig.tight_layout()

# %%
# test bp filtered

fig, axs = plt.subplots(ncols=len(files), nrows=2, sharex=True, figsize=(24, 10))

axs = axs.T

# filters:

# lowpass (for general use)
b_lp, a_lp = butter(N=2, Wn=50, btype="low", fs=fs)

# bandpass (for zero-point segmentation)
sos_bp = butter(N=2, Wn=[0.2, 50], btype="bandpass", fs=fs, output="sos")

for i_file, file in enumerate(files):

    breath = AudioObject.from_wav(file, channels=1, b=b_lp, a=a_lp)
    x = breath.get_x()

    # get lowpass filt
    breath_lowpass = breath.audio_filt

    # get bandpass filt (for segmentation)
    breath_bandpass = sosfiltfilt(
        sos_bp, breath.audio, padtype="constant", padlen=len(breath.audio) - 1
    )

    # plot lp waveform
    ax_lp = axs[i_file, 0]
    ax_lp.plot(x, breath_lowpass, linewidth=0.5)
    ax_lp.set(title=os.path.basename(file))

    # plot bp waveform
    ax_bp = axs[i_file, 1]
    ax_bp.plot(x, breath_bandpass, linewidth=0.5)

axs[0, 0].set(ylabel="amplitude")
axs[1, 1].set(xlabel="time since block onset (s)")
fig.tight_layout()

# %%
# test savitzky golay filtered

fig, axs = plt.subplots(ncols=len(files), nrows=2, sharex=True, figsize=(24, 10))
axs = axs.T

# lowpass (for general use)
b_lp, a_lp = butter(N=2, Wn=50, btype="low", fs=fs)

for i_file, file in enumerate(files):

    breath = AudioObject.from_wav(file, channels=1, b=b_lp, a=a_lp)
    x = breath.get_x()

    # get lowpass filt
    breath_lowpass = breath.audio_filt

    # savgol filter
    breath_bandpass = savgol_filter(
        breath.audio,
        window_length=int(10e-3 * fs),
        polyorder=6,
    )

    # plot lp waveform
    ax_lp = axs[i_file, 0]
    ax_lp.plot(x, breath_lowpass, linewidth=0.5)
    ax_lp.set(title=os.path.basename(file))

    # plot bp waveform
    ax_bp = axs[i_file, 1]
    ax_bp.plot(x, breath_bandpass, linewidth=0.5)

axs[0, 0].set(ylabel="amplitude")
axs[1, 1].set(xlabel="time since block onset (s)")
fig.tight_layout()

# %%
# test rolling median

window_length = int(0.05 * fs)

fig, axs = plt.subplots(ncols=len(files), nrows=2, sharex=True, figsize=(24, 10))
axs = axs.T

# lowpass
b_lp, a_lp = butter(N=2, Wn=50, btype="low", fs=fs)

for i_file, file in enumerate(files):
    breath = AudioObject.from_wav(file, channels=1, b=b_lp, a=a_lp)
    x = breath.get_x()

    # get lowpass filt
    breath_lowpass = breath.audio_filt

    rolling_median = pd.Series(breath_lowpass).rolling(window=window_length).median()
    breath_bandpass = np.array(breath_lowpass - rolling_median)

    # plot lp waveform
    ax_lp = axs[i_file, 0]
    ax_lp.plot(x, breath_lowpass, linewidth=0.5)
    ax_lp.set(title=os.path.basename(file))

    # plot bp waveform
    ax_bp = axs[i_file, 1]
    ax_bp.plot(x, breath_bandpass, linewidth=0.5)

axs[0, 0].set(ylabel="amplitude")
axs[1, 1].set(xlabel="time since block onset (s)")
fig.tight_layout()

# %%
# test rolling mean

window_length = int(1 * fs)

fig, axs = plt.subplots(ncols=len(files), nrows=2, sharex=True, figsize=(24, 10))
axs = axs.T

# lowpass
b_lp, a_lp = butter(N=2, Wn=50, btype="low", fs=fs)

for i_file, file in enumerate(files):
    breath = AudioObject.from_wav(file, channels=1, b=b_lp, a=a_lp)
    x = breath.get_x()

    # get lowpass filt
    breath_lowpass = breath.audio_filt

    rolling_mean = pd.Series(breath_lowpass).rolling(window=window_length).mean()
    breath_bandpass = np.array(breath_lowpass - rolling_mean)

    # plot lp waveform
    ax_lp = axs[i_file, 0]
    ax_lp.plot(x, breath_lowpass, linewidth=0.5)
    ax_lp.set(title=os.path.basename(file))

    # plot bp waveform
    ax_bp = axs[i_file, 1]
    ax_bp.plot(x, breath_bandpass, linewidth=0.5)

axs[0, 0].set(ylabel="amplitude")
axs[1, 1].set(xlabel="time since block onset (s)")
fig.tight_layout()

# %%
# test strict low-pass filter

window_length = int(0.3 * fs)

fig, axs = plt.subplots(ncols=len(files), nrows=2, sharex=True, figsize=(24, 10))
axs = axs.T


# lowpass
b_lp, a_lp = butter(N=2, Wn=50, btype="low", fs=fs)

# strict lowpass
b_slp, a_slp = butter(N=2, Wn=10, btype="low", fs=fs)

for i_file, file in enumerate(files):
    breath = AudioObject.from_wav(file, channels=1, b=b_lp, a=a_lp)
    x = breath.get_x()

    # get lowpass filt
    breath_lowpass = breath.audio_filt

    rolling_median = np.array(
        pd.Series(breath_lowpass).rolling(window=window_length).median()
    )
    # backfill first window with first non-nan value
    i_first_non_nan = int(np.flatnonzero(~np.isnan(rolling_median))[0])
    rolling_median[:i_first_non_nan] = rolling_median[i_first_non_nan]

    # breath_seg = breath_lowpass - rolling_median
    breath_seg = filtfilt(b_slp, a_slp, breath_lowpass - rolling_median)

    exps, insps = segment_breaths(
        breath_seg,
        do_filter=False,
        threshold_exp=lambda x: np.percentile(x, 30),
        threshold_insp=lambda x: np.percentile(x, 30),
        fs=None,
    )

    # plot lp waveform
    ax_lp = axs[i_file, 0]
    ax_lp.plot(x, breath_lowpass, linewidth=0.5)
    ax_lp.set(title=os.path.basename(file))
    ax_lp.scatter(
        x[exps],
        breath_lowpass[exps],
        **plot_kwargs_exp,
    )
    ax_lp.scatter(
        x[insps],
        breath_lowpass[insps],
        **plot_kwargs_insp,
    )
    ax_lp.grid()

    # plot bp waveform
    ax_bp = axs[i_file, 1]
    ax_bp.plot(x, breath_seg, linewidth=0.5)
    ax_bp.scatter(
        x[exps],
        breath_seg[exps],
        **plot_kwargs_exp,
    )
    ax_bp.scatter(
        x[insps],
        breath_seg[insps],
        **plot_kwargs_insp,
    )
    ax_bp.grid()

axs[0, 0].set(ylabel="amplitude")
axs[1, 1].set(xlabel="time since block onset (s)")
fig.tight_layout()

# %%
# rolling median: amplitude distributions

window_length = int(0.7 * fs)
bw = 50  # 100

fig_wave, axs_wave = plt.subplots(
    ncols=len(files), nrows=2, sharex=True, figsize=(24, 10)
)
axs_wave = axs_wave.T

fig_dist, axs_dist = plt.subplots(
    ncols=len(files), nrows=2, sharex=True, figsize=(24, 10)
)
axs_dist = axs_dist.T

# lowpass
b_lp, a_lp = butter(N=2, Wn=50, btype="low", fs=fs)

for i_file, file in enumerate(files):
    breath = AudioObject.from_wav(file, channels=1, b=b_lp, a=a_lp)
    x = breath.get_x()

    # get lowpass filt
    breath_lowpass = breath.audio_filt

    rolling_median = np.array(
        pd.Series(breath_lowpass).rolling(window=window_length).median()
    )
    # backfill first window with first non-nan value
    i_first_non_nan = int(np.flatnonzero(~np.isnan(rolling_median))[0])
    rolling_median[:i_first_non_nan] = rolling_median[i_first_non_nan]
    breath_bandpass = breath_lowpass - rolling_median

    # plot lp waveform
    ax_lp = axs_wave[i_file, 0]
    ax_lp.plot(x, breath_lowpass, linewidth=0.5)
    ax_lp.plot(x, rolling_median, linewidth=0.5)
    ax_lp.set(title=os.path.basename(file))

    # plot bp waveform
    ax_bp = axs_wave[i_file, 1]
    ax_bp.plot(x, breath_bandpass, linewidth=0.5)

    # plot distributions
    ax_lp_dist = axs_dist[i_file, 0]
    ax_lp_dist.set(title=os.path.basename(file))
    plot_amplitude_dist(breath_lowpass, ax_lp_dist, binwidth=bw)

    ax_bp_dist = axs_dist[i_file, 1]
    plot_amplitude_dist(breath_bandpass, ax_bp_dist, binwidth=bw)

    # plot fitted distribution
    x_dist = np.linspace(breath_bandpass.min(), breath_bandpass.max(), 100)
    kde = gaussian_kde(breath_bandpass)
    p = kde(x_dist)

    # get indices of highest 2 peaks - should be insp/exp levels
    peaks = find_peaks(p)[0]
    top2 = sorted(peaks[np.argsort(p[peaks])][-2:])

    trough = top2[0] + np.argmin(p[np.arange(*top2)])

    ax_bp_dist.plot(x_dist, p, color="k")
    ax_bp_dist.scatter(x_dist[top2], p[top2], color="r", marker="+", s=16, zorder=3)
    ax_bp_dist.scatter(x_dist[trough], p[trough], color="r", marker="+", s=16, zorder=3)


axs_wave[0, 0].set(ylabel="amplitude")
axs_wave[1, 1].set(xlabel="time since block onset (s)")
fig_wave.tight_layout()

axs_dist[0, 0].set(ylabel="density")
axs_dist[1, 1].set(xlabel="amplitude")
fig_dist.tight_layout()

# %%
# threshold from amplitude distr fit

window_length = int(0.7 * fs)
bw = 50

fig_wave, axs_wave = plt.subplots(
    ncols=len(files), nrows=2, sharex=True, figsize=(24, 10)
)
axs_wave = axs_wave.T

fig_dist, axs_dist = plt.subplots(
    ncols=len(files), nrows=2, sharex=True, figsize=(24, 10)
)
axs_dist = axs_dist.T

# lowpass
b_lp, a_lp = butter(N=2, Wn=50, btype="low", fs=fs)

for i_file, file in enumerate(files):
    breath = AudioObject.from_wav(file, channels=1, b=b_lp, a=a_lp)
    x = breath.get_x()

    # get lowpass filt
    breath_lowpass = breath.audio_filt

    rolling_median = np.array(
        pd.Series(breath_lowpass).rolling(window=window_length).median()
    )
    # backfill first window with first non-nan value
    i_first_non_nan = int(np.flatnonzero(~np.isnan(rolling_median))[0])
    rolling_median[:i_first_non_nan] = rolling_median[i_first_non_nan]
    breath_bandpass = breath_lowpass - rolling_median

    # plot distributions
    ax_lp_dist = axs_dist[i_file, 0]
    ax_lp_dist.set(title=os.path.basename(file))
    plot_amplitude_dist(breath_lowpass, ax_lp_dist, binwidth=bw)

    ax_bp_dist = axs_dist[i_file, 1]
    plot_amplitude_dist(breath_bandpass, ax_bp_dist, binwidth=bw)

    # plot fitted distribution
    x_dist = np.linspace(breath_bandpass.min(), breath_bandpass.max(), 100)
    kde = gaussian_kde(breath_bandpass)
    p = kde(x_dist)

    # get indices of highest 2 peaks - should be insp/exp levels
    peaks = find_peaks(p)[0]
    top2 = sorted(peaks[np.argsort(p[peaks])][-2:])

    trough = top2[0] + np.argmin(p[np.arange(*top2)])

    ax_bp_dist.plot(x_dist, p, color="k")
    ax_bp_dist.scatter(x_dist[top2], p[top2], color="r", marker="+", s=16, zorder=3)
    ax_bp_dist.scatter(x_dist[trough], p[trough], color="r", marker="+", s=16, zorder=3)
    
    # get insps/exps
    exps, insps = segment_breaths(
        breath_bandpass,
        do_filter=False,
        threshold = lambda x: x_dist[trough],
        fs=None,
    )
    # plot lp waveform
    ax_lp = axs_wave[i_file, 0]
    ax_lp.plot(x, breath_lowpass, linewidth=0.5)
    ax_lp.plot(x, rolling_median, linewidth=0.5)
    ax_lp.set(title=os.path.basename(file))
    ax_lp.scatter(
        x[exps],
        breath_lowpass[exps],
        **plot_kwargs_exp,
    )
    ax_lp.scatter(
        x[insps],
        breath_lowpass[insps],
        **plot_kwargs_insp,
    )

    # plot bp waveform
    ax_bp = axs_wave[i_file, 1]
    ax_bp.plot(x, breath_bandpass, linewidth=0.5)
    ax_bp.scatter(
        x[exps],
        breath_bandpass[exps],
        **plot_kwargs_exp,
    )
    ax_bp.scatter(
        x[insps],
        breath_bandpass[insps],
        **plot_kwargs_insp,
    )


axs_wave[0, 0].set(ylabel="amplitude")
axs_wave[1, 1].set(xlabel="time since block onset (s)")
fig_wave.tight_layout()

axs_dist[0, 0].set(ylabel="density")
axs_dist[1, 1].set(xlabel="amplitude")
fig_dist.tight_layout()

# %%
# threshold from amplitude distr fit - rolling min subtraction

# ideally at breathing rate ,so you always normalize to most recent insp
window_length = int(0.5 * fs)
bw = 50

fig_wave, axs_wave = plt.subplots(
    ncols=len(files), nrows=2, sharex=True, figsize=(24, 10)
)
axs_wave = axs_wave.T

fig_dist, axs_dist = plt.subplots(
    ncols=len(files), nrows=2, sharex=True, figsize=(24, 10)
)
axs_dist = axs_dist.T

# lowpass
b_lp, a_lp = butter(N=2, Wn=50, btype="low", fs=fs)

for i_file, file in enumerate(files):
    breath = AudioObject.from_wav(file, channels=1, b=b_lp, a=a_lp)
    x = breath.get_x()

    # get lowpass filt
    breath_lowpass = breath.audio_filt

    rolling_min = np.array(
        pd.Series(breath_lowpass).rolling(window=window_length).min()
    )
    # backfill first window with first non-nan value
    i_first_non_nan = int(np.flatnonzero(~np.isnan(rolling_min))[0])
    rolling_min[:i_first_non_nan] = rolling_min[i_first_non_nan]
    breath_bandpass = breath_lowpass - rolling_min

    # plot distributions
    ax_lp_dist = axs_dist[i_file, 0]
    ax_lp_dist.set(title=os.path.basename(file))
    plot_amplitude_dist(breath_lowpass, ax_lp_dist, binwidth=bw)

    ax_bp_dist = axs_dist[i_file, 1]
    plot_amplitude_dist(breath_bandpass, ax_bp_dist, binwidth=bw)

    # plot fitted distribution
    x_dist = np.linspace(breath_bandpass.min(), breath_bandpass.max(), 100)
    kde = gaussian_kde(breath_bandpass)
    p = kde(x_dist)

    
    peaks = find_peaks(p)[0]

    # push closest peak to 0; should be the case with rolling min subtraction
    if p[0] > p[min(peaks)]:
        peaks[peaks.argmin()] = 0

    top2 = sorted(peaks[np.argsort(p[peaks])][-2:]) # get indices of highest 2 peaks.

    trough = top2[0] + np.argmin(p[np.arange(*top2)])

    ax_bp_dist.plot(x_dist, p, color="k")
    ax_bp_dist.scatter(x_dist[top2], p[top2], color="r", marker="+", s=16, zorder=3)
    ax_bp_dist.scatter(x_dist[trough], p[trough], color="r", marker="+", s=16, zorder=3)
    
    # get insps/exps
    exps, insps = segment_breaths(
        breath_bandpass,
        do_filter=False,
        threshold = lambda x: x_dist[trough],
        fs=None,
    )
    # plot lp waveform
    ax_lp = axs_wave[i_file, 0]
    ax_lp.plot(x, breath_lowpass, linewidth=0.5)
    ax_lp.plot(x, rolling_min, linewidth=0.5, color=[1,1,0])
    ax_lp.set(title=os.path.basename(file))
    ax_lp.scatter(
        x[exps],
        breath_lowpass[exps],
        **plot_kwargs_exp,
    )
    ax_lp.scatter(
        x[insps],
        breath_lowpass[insps],
        **plot_kwargs_insp,
    )

    # plot bp waveform
    ax_bp = axs_wave[i_file, 1]
    ax_bp.plot(x, breath_bandpass, linewidth=0.5) 
    ax_bp.scatter(
        x[exps],
        breath_bandpass[exps],
        **plot_kwargs_exp,
    )
    ax_bp.scatter(
        x[insps],
        breath_bandpass[insps],
        **plot_kwargs_insp,
    )


axs_wave[0, 0].set(ylabel="amplitude")
axs_wave[1, 1].set(xlabel="time since block onset (s)")
fig_wave.tight_layout()

axs_dist[0, 0].set(ylabel="density")
axs_dist[1, 1].set(xlabel="amplitude")
fig_dist.tight_layout()