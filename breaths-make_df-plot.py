# %%
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 1
# %matplotlib widget
# %aimport utils.audio
# %aimport utils.breath
# %aimport utils.callbacks

from utils.audio import AudioObject
from utils.breath import (
    segment_breaths,
    make_notmat_vars,
    plot_breath_callback_trial,
    get_kde_threshold,
)
from utils.callbacks import call_mat_stim_trial_loader
from utils.evfuncs import segment_notes
from utils.file import parse_birdname, parse_parameter_from_string
from utils.json import merge_json, NumpyEncoder
from utils.video import get_triggers_from_audio

# %%
# get filelist

paths = [
    r"M:/eszter/behavior/air sac calls/HVC lesion/aspiration/rd99rd72/preLesion/callbacks/rand/230215/*-B*.wav",
    r"M:/eszter/behavior/air sac calls/HVC lesion/aspiration/pk19br8/preLesion/callback/rand/**/*-B*.wav",
    # r"M:/eszter/behavior/air sac calls/HVC lesion/aspiration/rd56/preLesion/callbacks/male_230117/*-B*.wav",  # this bird has wonky amplitude distributions
    r"M:/eszter/behavior/air sac calls/HVC lesion/aspiration/rd57rd97/preLesion/callbacks/male_230117/*-B*.wav",
]

default_bird = None

json_filename = r".\data\breath_figs\plot_metadata.json"

# `*-B*` excludes "-PostBlock"

# get all files matching above paths
files = [file for path in paths for file in sorted(glob.glob(os.path.normpath(path)))]

assert len(files) != 0, "No files found!"

print("Files: ")
for i, f in enumerate(files):
    print(f"{i}. {os.path.split(f)[1]}")

# %%


# construct trial-by-trial df across all files
all_trials = []
all_breaths = []

fs = 44100
b, a = butter(N=2, Wn=50, btype="low", fs=fs)

def check_call(trial, breath_norm, threshold, fs):
    """
    pass in a row of `stim_trials` to check whether there was a call in file 
    """
    ii = np.array([trial["trial_start_s"], trial["trial_end_s"]]) * fs
    ii[1] = min(ii[1], len(breath_norm))  # account for trial_end_s == np.inf
    ii = ii.astype(int)

    putative_call = (breath_norm[ii[0] : ii[1]].max() >= threshold)

    return putative_call


for f in files:

    # load audio
    channels = AudioObject.from_wav(
        f, channels="all", channel_names=["audio", "breathing", "trigger"]
    )

    assert fs == channels[1].fs, "Wrong sample rate!"

    channels[1].filtfilt(b, a)  # filter breathing
    breath = channels[1].audio_filt

    # threshold stimuli; assume 100ms length
    stims = get_triggers_from_audio(channels[2].audio, crossing_direction="down") / fs

    # segment breaths based on smoothed waveform
    breath_zero_point = get_kde_threshold(breath)
    centering = lambda x: breath_zero_point
    exps, insps = segment_breaths(breath, channels[1].fs, threshold=centering, b=b, a=a)

    # get onsets, offsets, labels
    onsets, offsets, labels = make_notmat_vars(
        exps, insps, len(breath), exp_label="exp", insp_label="insp"
    )
    onsets = onsets / fs
    offsets = offsets / fs

    # mimic .not.mat format
    data = {
        "onsets": np.concatenate([onsets, stims]) * 1000,
        "offsets": np.concatenate([offsets, stims + 0.1]) * 1000,
        "labels": np.concatenate(
            [labels, ["Stimulus"] * len(stims)]
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

    calls["wav_filename"] = f
    stim_trials["wav_filename"] = f
    stim_trials["breath_zero_point"] = breath_zero_point

    # putative stim phase: for now just "exp" or "insp", based on first "call"
    stim_trials["stim_phase"] = stim_trials["calls_in_range"].apply(
        lambda x: calls.loc[x[0], "type"] if len(x) > 0 else "error"
    )

    # putative calls: relative to deepest insp in file.
    breath_norm = (breath - breath_zero_point) / np.abs(breath.min())

    stim_trials["putative_call"] = stim_trials.apply(
        check_call,
        breath_norm=breath_norm,
        threshold=1.1,
        fs=fs,
        axis=1,
    )

    all_trials.append(
        stim_trials.reset_index().set_index(["wav_filename", "stims_index"])
    )

    all_breaths.append(calls.reset_index().set_index(["wav_filename", "calls_index"]))

all_trials = pd.concat(all_trials).sort_index()
all_breaths = pd.concat(all_breaths).sort_index()

all_trials

# %%
# pickle all_trials
figure_root_dir = "./data/breath_figs-spline_fit"

with open(os.path.join(figure_root_dir, "all_trials.pickle"), "wb") as f:
    pickle.dump(all_trials, f)

with open(os.path.join(figure_root_dir, "all_breaths.pickle"), "wb") as f:
    pickle.dump(all_breaths, f)

# %%
# plot options

exist_ok = True  # False --> error out if folder already exists
skip_replot = False  # True --> if the plot path already exists, skip replot (just process metadata)

pre_time_s = 0.1
post_time_s = 3.1
ylims = [-3500, 10000]

# %%
# plot & make json records

records = {}
for file in all_trials.index.get_level_values("wav_filename").unique():
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

        breath_filt = filtfilt(b, a, ao.audio)

        fig, ax = plt.subplots(figsize=(10, 5))

        if not os.path.exists(plot_filename) or not skip_replot:
            ax = plot_breath_callback_trial(
                breath=breath_filt,
                fs=ao.fs,
                stim_trial=stim_trials.loc[t],
                y_breath_labels=metadata["breath_zero_point"],
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
# dump json records

# get old records
if os.path.exists(json_filename):
    with open(json_filename, "r") as jf:
        extant_records = {x["plot_id"]: x for x in json.loads(jf.read())}
else:
    extant_records = {}

extant_records = merge_json(
    records,
    extant_records,
    dict_fields={"plot_filename": "kde-threshold"},
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
