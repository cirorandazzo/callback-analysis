# %%
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt  

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # attempt to prevent spectrogram-related memory leak :(

# %load_ext autoreload
# %autoreload 1
# %matplotlib widget
# %aimport utils.audio
# %aimport utils.breath
# %aimport utils.callbacks

from utils.audio import AudioObject
from utils.callbacks import call_mat_stim_trial_loader
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

fs = 44100
b, a = butter(N=2, Wn=50, btype="low", fs=fs)

for f in files:

    # load audio
    channels = AudioObject.from_wav(
        f, channels="all", channel_names=["audio", "breathing", "trigger"]
    )

    assert fs == channels[1].fs, "Wrong sample rate!"

    channels[0].filtfilt(b, a)  # filter breathing
    audio = channels[0].audio_filt

    # threshold stimuli; assume 100ms length
    stims = get_triggers_from_audio(channels[2].audio, crossing_direction="down") / fs

    data = {
        "onsets": stims * 1000,
        "offsets": (stims + 0.1) * 1000,
        "labels": np.array(["Stimulus"] * len(stims)),
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

    all_trials.append(
        stim_trials.reset_index().set_index(["wav_filename", "stims_index"])
    )

all_trials = pd.concat(all_trials).sort_index()

all_trials

# %%
# plot options

exist_ok = True  # False --> error out if folder already exists
skip_replot = True  # True --> if the plot path already exists, skip replot (just process metadata)

pre_time_s = 0.1
post_time_s = 3.1
ylims = [0, 15e3]  # Hz
figure_root_dir = "./data/spectrograms"
spectrograms_root_dir = "./data/audio_objs"

# %%
# pickle all_trials

with open(os.path.join(figure_root_dir, "all_trials.pickle"), "wb") as f:
    pickle.dump(all_trials, f)

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

    audiofile_plot_folder = os.path.join(figure_root_dir, bird, basename)
    os.makedirs(audiofile_plot_folder, exist_ok=exist_ok)

    spx_path = os.path.join(spectrograms_root_dir, bird)
    os.makedirs(spx_path, exist_ok=exist_ok)

    block = int(parse_parameter_from_string(root, "Block", chars_to_ignore=0))

    # load audio & relevant stims
    stim_trials = all_trials.xs(file)

    pickled_audio = os.path.join(spx_path, f"{basename}.pickle")
    if os.path.exists(pickled_audio):
        pass
        # with open( pickled_audio, "rb") as f:
        #     ao = pickle.load(f)
    else:
        ao = AudioObject.from_wav(file, channels=0)
        ao.filtfilt_butter_default(f_low=500, f_high=15e3)
        ao.make_spectrogram()

        with open( pickled_audio, "wb") as f:
            pickle.dump(ao, f)

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
            ]
        }
        metadata["wav_filename"] = file
        metadata["trial"] = t
        metadata["bird"] = bird
        metadata["block"] = block
        metadata["plot_id"] = f"{basename}-tr{t}"
        metadata["plot_filename"] = plot_filename

        st = metadata["trial_start_s"]
        en = metadata["trial_end_s"]

        if np.isinf(metadata["trial_end_s"]):
            metadata["trial_end_s"] = "Inf"

        if not os.path.exists(plot_filename) or not skip_replot:

            fig, ax = plt.subplots(figsize=(10, 5))

            ao.plot_spectrogram(x_offset_s=-1*st, ax=ax, vmin=0.7)

            ax.set(
                title=metadata["plot_id"],
                xlim=[-1 * pre_time_s, post_time_s],
                xlabel="Time, stim-aligned (s)",
                ylim=ylims,
            )

            fig.savefig(plot_filename)
            plt.close("all")

        records[metadata["plot_id"]] = metadata

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
    dict_fields={"plot_filename": "spectrogram"},
    fields_to_remove=("breath_zero_point", "calls_index"),
    keep_extant_fields=True,
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
