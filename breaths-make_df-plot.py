# %%
import glob
import json
import os

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
from utils.breath import segment_breaths, make_notmat_vars, plot_breath_callback_trial
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

default_bird = "rd56"  # only bird which fails name parsing

# `*-B*` excludes "-PostBlock"

# get all files matching above paths
files = [file for path in paths for file in sorted(glob.glob(os.path.normpath(path)))]

assert len(files) != 0, "No files found!"

print("Files: ")
for i, f in enumerate(files):
    print(f"{i}. {os.path.split(f)[1]}")

# %%
# construct trial-by-trial df across all files
all_trials = pd.DataFrame()

fs = 44100
b, a = butter(N=2, Wn=50, btype="low", fs=fs)

for f in files:

    # load audio
    channels = AudioObject.from_wav(
        f, channels="all", channel_names=["audio", "breathing", "trigger"]
    )

    assert fs == channels[1].fs
    breath = channels[1].audio

    # threshold stimuli; assume 100ms length
    stims = get_triggers_from_audio(channels[2].audio, crossing_direction="down") / fs

    # segment breaths based on smoothed waveform
    centering = lambda x: np.percentile(x, 45)
    exps, insps = segment_breaths(breath, channels[1].fs, threshold=centering, b=b, a=a)
    breath_zero_point = centering(filtfilt(b, a, breath))

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

    all_trials = pd.concat(
        [
            all_trials,
            stim_trials.reset_index().set_index(["wav_filename", "stims_index"]),
        ]
    )


all_trials

# %%
# plot & make json records

exist_ok = True  # False --> error out if folder already exists
skip_replot = True  # True --> if the plot path already exists, skip replot (just process metadata)

pre_time_s = 0.1
post_time_s = 3.1
ylims = [-3500, 10000]
figure_root_dir = "./data/breath_figs"

records = {}
for file in np.unique(all_trials.index.get_level_values("wav_filename")):
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
        plot_filename = os.path.normpath(os.path.join(audiofile_plot_folder, f"tr{t}.jpg"))

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
                audio=breath_filt,
                fs=ao.fs,
                stim_trial=stim_trials.loc[t],
                y_breath_labels=metadata["breath_zero_point"],
                pre_time_s=pre_time_s,
                post_time_s=post_time_s,
                ylims=ylims,
                st=st,
                en=en,
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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle numpy scalar types (e.g., np.int64, np.float64)
        if isinstance(obj, np.generic):
            return obj.item()  # Convert to native Python type (e.g., int, float)

        # Handle numpy arr` ays (convert to a list)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        # Fallback to the default method for other types
        return super().default(obj)


json_filename = os.path.join(figure_root_dir, "plot_metadata.json")

# get old records
if os.path.exists(json_filename):
    with open(json_filename, "r") as jf:
        extant_records = {x["plot_id"]: x for x in json.loads(jf.read())}
else:
    extant_records = {}

# adds new records, overwriting extant records with same plot id
for k, v in records.items():
    extant_records[k] = v

# write records
with open(json_filename, "w") as f:
    # sorts by key before dumping
    json.dump(
        [extant_records[k] for k in sorted(extant_records.keys())],
        f,
        indent=4,
        cls=NumpyEncoder,
    )
