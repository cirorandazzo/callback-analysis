# %%
#

import json
import glob
import os
import pickle

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt

from utils.audio import AudioObject
from utils.breath import make_notmat_vars, plot_breath_callback_trial, segment_breaths
from utils.callbacks import call_mat_stim_trial_loader
from utils.evfuncs import segment_notes
from utils.file import parse_birdname, parse_parameter_from_string
from utils.json import merge_json, NumpyEncoder
from utils.video import get_triggers_from_audio

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

figure_save_folder = "./data/rolling_min-multi"

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


def make_rolling_min_plot(
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
    binwidth=50,
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


# %%
# PROCESS DATA

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

    fig_multiplot, axs_multiplot = make_rolling_min_plot(**data_dict)
    axs_multiplot.ravel()[0].set(title=basename)

    fig_multiplot.savefig(fig_savepath)


# %%

pickle_file = os.path.join(figure_save_folder, "rolling_min_subtracted.pickle")

df = pd.DataFrame.from_records(processed_data).set_index("i_file")

with open(pickle_file, "wb") as f:
    pickle.dump(df, f)

print(f"Successfully dumped data to: {pickle_file}")

# %%

df

# %%
# MAKE TRIAL-INDEXED DF

all_trials = []

for i_file in df.index:

    f, breath, exps, insps = [
        df.loc[i_file, col]
        for col in ("file", "breath_roll_min_subtr", "exps", "insps")
    ]

    # load audio
    channels = AudioObject.from_wav(
        f, channels="all", channel_names=["audio", "breathing", "trigger"]
    )

    assert fs == channels[1].fs

    # threshold stimuli; assume 100ms length
    stims = get_triggers_from_audio(channels[2].audio, crossing_direction="down") / fs

    # segment breaths based on smoothed waveform
    centering = lambda x: np.percentile(x, 45)
    breath_zero_point = centering(breath)

    # get onsets, offsets, labels
    onsets, offsets, labels = make_notmat_vars(
        exps, insps, len(breath), exp_label="exp", insp_label="insp"
    )
    onsets = onsets / fs
    offsets = offsets / fs

    calls_on, calls_off = segment_notes(
        smooth=breath, fs=fs, min_int=10, min_dur=0, threshold=np.percentile(breath, 98)
    )

    # mimic .not.mat format
    data = {
        "onsets": np.concatenate([onsets, calls_on, stims]) * 1000,
        "offsets": np.concatenate([offsets, calls_off, stims + 0.1]) * 1000,
        "labels": np.concatenate(
            [labels, ["call"] * len(calls_on), ["Stimulus"] * len(stims)]
        ),
    }

    calls, stim_trials, rejected_trials, file_info, call_types = (
        call_mat_stim_trial_loader(
            file=None,
            data=data,
            from_notmat=True,
            verbose=False,
            acceptable_call_labels=["Stimulus", "exp", "insp", "call"],
        )
    )

    stim_trials["wav_filename"] = f
    stim_trials["breath_zero_point"] = breath_zero_point

    # putative stim phase: for now just "exp" or "insp", based on first "call"
    stim_trials["stim_phase"] = stim_trials["calls_in_range"].apply(
        lambda x: calls.loc[x[0], "type"] if len(x) > 0 else "error"
    )

    # putative calls: based on amplitude segmentation
    putative_calls = list(calls.loc[calls["type"] == "call"].index)
    stim_trials["putative_call"] = stim_trials["calls_in_range"].apply(
        lambda x: any([y in putative_calls for y in x])
    )

    all_trials.append(
        stim_trials.reset_index().set_index(["wav_filename", "stims_index"]),
    )

all_trials = pd.concat(all_trials)  # concat file by file df

all_trials

# %%
# PICKLE TRIAL-INDEXED DF

pickle_file = os.path.join(figure_save_folder, "rolling_min_subtracted-by_trial.pickle")

with open(pickle_file, "wb") as f:
    pickle.dump(all_trials, f)

print(f"Successfully dumped data to: {pickle_file}")

# %%
# PICKLE LOAD ALL_TRIALS & DF

with open(r".\data\rolling_min-multi\rolling_min_subtracted-by_trial.pickle", "rb") as f:
    all_trials = pickle.load(f)

with open(r".\data\rolling_min-multi\rolling_min_subtracted.pickle", "rb") as f:
    df = pickle.load(f)

# %%
# plot & make json records

exist_ok = True  # False --> error out if folder already exists
skip_replot = True  # True --> if the plot path already exists, skip replot (just process metadata)

pre_time_s = 0.1
post_time_s = 3.1
ylims = [-3500, 10000]
figure_root_dir = "./data/rolling_min_seg-filtered_trace"

records = {}
for file in all_trials.index.get_level_values("wav_filename").unique():
    i_file = np.flatnonzero(df["file"] == file)
    assert (
        len(i_file) == 1
    ), f"Must be exactly one row in df with this filename! Found {len(i_file)}. Filename: {file}."
    i_file = i_file[0]

    # paths
    root = os.path.splitext(file)[0]  # entire filename without extension
    basename = os.path.split(root)[-1]

    # metadata
    try:
        bird = parse_birdname(root)
        # bird = "rd56"
    except TypeError:
        if default_bird is not None:
            bird = default_bird
        else:
            raise TypeError(
                f"Couldn't parse birdname from: {root}\nIf this bird just has one band, you should find this and hard-code its name."
            )

    block = int(parse_parameter_from_string(root, "Block", chars_to_ignore=0))

    # load audio & relevant stims
    stim_trials = all_trials.xs(file)
    ao = AudioObject.from_wav(file, channels=1)

    audiofile_plot_folder = os.path.join(figure_root_dir, bird, basename)
    os.makedirs(audiofile_plot_folder, exist_ok=exist_ok)

    # for each stim trial in this audio file
    for t in stim_trials.index:
        # make metadata for json
        plot_filename = os.path.normpath(
            os.path.join(audiofile_plot_folder, f"tr{t}.jpg")
        )

        metadata = {
            k: stim_trials.loc[t, k]
            for k in [
                "trial_start_s",
                "trial_end_s",
                "calls_index",
                "stim_phase",
                "putative_call",
                "breath_zero_point",
            ]
        }
        metadata["wav_filename"] = file
        metadata["trial"] = t
        metadata["bird"] = bird
        metadata["block"] = block
        metadata["call_types"] = np.unique(stim_trials.loc[t, "call_types"])
        metadata["plot_id"] = f"{basename}-tr{t}"
        metadata["plot_filename"] = plot_filename

        st = metadata["trial_start_s"]
        en = metadata["trial_end_s"]

        if np.isinf(metadata["trial_end_s"]):
            metadata["trial_end_s"] = "Inf"

        breath = df.loc[i_file, "breath_lowpass"]

        fig, ax = plt.subplots(figsize=(10, 5))

        if not os.path.exists(plot_filename) or not skip_replot:
            ax = plot_breath_callback_trial(
                breath=breath,
                fs=ao.fs,
                stim_trial=stim_trials.loc[t],
                y_breath_labels="infer",
                pre_time_s=pre_time_s,
                post_time_s=post_time_s,
                ylims=ylims,
                st_s=st,
                en_s=en,
                ax=ax,
                color_dict={"exp": "r", "insp": "b"},
            )

            ax.set(title=metadata["plot_id"])

            fig.savefig(plot_filename)
        records[metadata["plot_id"]] = metadata
        plt.close()

records

# %%
# MERGE JSONS

json_filename = "./data/breath_figs/plot_metadata.json"

# get old records
if os.path.exists(json_filename):
    with open(json_filename, "r") as jf:
        extant_records = {x["plot_id"]: x for x in json.loads(jf.read())}
else:
    extant_records = {}

extant_records = merge_json(
    records,
    extant_records,
    dict_fields={"plot_filename" : "lowpass_trace-rolling_min_seg"},
    fields_to_remove=("breath_zero_point", "calls_index"),
)

# write records
with open(json_filename, "w") as f:
    # sorts by key before dumping
    json.dump(
        [extant_records[k] for k in sorted(extant_records.keys())],
        f,
        indent=4,
        cls=NumpyEncoder,
    )
