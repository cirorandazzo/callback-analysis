# %%
# insp categories: all breaths
#
# insp_categories.py uses only first insp, from dataframe all_trials
# this will use all breaths, from dataframe all_breaths
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
import time

import numpy as np
import pandas as pd
from scipy.signal import butter
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt

import umap

plt.rcParams.update({"savefig.dpi": 400})

from utils.audio import AudioObject
from utils.breath import get_first_breath_segment
from utils.file import parse_birdname

# %%
# load `all_trials` and `all_breaths` dataframe

fs = 44100

with open("./data/breath_figs-spline_fit/all_trials.pickle", "rb") as f:
    all_trials = pickle.load(f)

with open("./data/breath_figs-spline_fit/all_breaths.pickle", "rb") as f:
    all_breaths = pickle.load(f)

# reject stimuli, which are also stored in this df
stims = all_breaths.loc[all_breaths["type"] == "Stimulus"]
all_breaths = all_breaths.loc[ all_breaths["type"].apply(lambda x: x in ["insp", "exp"])]

all_breaths

# %%
# get trace, amplitude for all breaths

# filter
b, a = butter(N=2, Wn=50, btype="low", fs=fs)

def get_breath_seg(start_s, end_s, breath, fs):

    ii_audio = (np.array([start_s, end_s]) * fs).astype(int)
    cut_breath = breath[np.arange(*ii_audio)]

    return cut_breath


for wav_filename, file_breaths in all_breaths.groupby("wav_filename"):

    zero_point = all_trials.xs(level="wav_filename", key=wav_filename)["breath_zero_point"].unique()
    assert len(zero_point) == 1
    zero_point = zero_point[0]

    breath = AudioObject.from_wav(wav_filename, channels=1, b=b, a=a).audio_filt

    #  map [biggest_insp, zero_point] --> [-1, 0]
    breath -= zero_point
    breath /= np.abs(breath.min())

    all_breaths.loc[file_breaths.index, "breath_norm"] = file_breaths.apply(
        lambda x: get_breath_seg(*x[["start_s", "end_s"]], breath, fs),
        axis=1,
    )

all_breaths["amplitude"] = all_breaths["breath_norm"].apply(lambda x: np.abs(x).max())

# %%
# assert previous call_index within this file corresponds to previous breath segment

allowed = 0.001

for wav_filename, file_breaths in all_breaths.groupby("wav_filename"):

    for idx, row in file_breaths.reset_index().iterrows():
        if row["calls_index"] == 0:
            # first call in file
            pass

        else:
            prev = all_breaths.loc[(wav_filename, row["calls_index"] - 1)]

            # correct timing
            assert (row["start_s"] - prev["end_s"]) < allowed

            # correct type
            assert row["type_prev_call"] == prev["type"]


# %%
# get putative_calls

# obvious on exp: amplitude threshold

all_breaths["putative_call"] = False

threshold = 1.1

ii_call_exps = (all_breaths["type"] == "exp") & (all_breaths["amplitude"] >= threshold)
all_breaths.loc[ii_call_exps, "putative_call"] = True


for idx, breath in all_breaths[ii_call_exps].iterrows():

    if idx[1] == 0:
        # first breath of file is call
        pass

    else:
        # insp just before
        all_breaths.loc[(idx[0], idx[1] - 1), "putative_call"] = True


# %%
# add stim/trial info to all_breaths


def add_trial_info(all_breaths, all_trials):

    new_columns = ["stims_index", "trial_index"]

    assert not all([c in all_breaths.reset_index().columns for c in new_columns]), f"Trial info already in all_breaths!"

    # Initialize with nullable Int16 dtype
    all_breaths["stims_index"] = pd.Series(dtype=pd.Int16Dtype())
    all_breaths["trial_index"] = pd.Series(dtype=pd.Int16Dtype())

    all_trials = all_trials.sort_values(["wav_filename", "trial_start_s"])

    # Iterate over each wav_filename group
    for wav_filename, breaths_group in all_breaths.groupby(level="wav_filename"):
        trials_group = (
            all_trials.loc[wav_filename] if wav_filename in all_trials.index else None
        )

        if trials_group is not None:
            # Assign stims_index based on trial start time <= breath end_s
            for i_breath, breath in breaths_group.iterrows():
                # Find the most recent trial that started before the breath's end time
                matching_trials = trials_group[
                    trials_group["trial_start_s"] <= breath["end_s"]
                ]
                if not matching_trials.empty:
                    # Last matching trial
                    all_breaths.at[i_breath, "stims_index"] = matching_trials.index[-1]
            # Assign trial_index within each trial
            for stims_index, trial in trials_group.iterrows():
                trial_breaths = breaths_group[
                    (breaths_group["end_s"] >= trial["trial_start_s"])
                    & (breaths_group["end_s"] <= trial["trial_end_s"])
                ]
                all_breaths.loc[trial_breaths.index, "trial_index"] = range(
                    len(trial_breaths)
                )

    return all_breaths

all_breaths = add_trial_info(all_breaths, all_trials)

all_breaths


# %%
# interpolated

# interpolate_length = int(12e3)  # 96.9% of trials shorter than this.
interpolate_length = int(15253)  # used for first-insp umap embeddings

all_breaths["breath_interpolated"] = all_breaths["breath_norm"].apply(
    lambda trial: np.interp(
        np.linspace(0, len(trial), interpolate_length),
        np.arange(len(trial)),
        trial,
    )
)


# %%
# prep umap parameters

save_folder = pathlib.Path("M:\public\Ciro\callback-breaths\umap-all_breaths")

metrics=[
        "cosine",
        # "correlation",
        "euclidean",
    ]

datasets = {
    k: np.vstack(all_breaths.loc[all_breaths["type"] == k, "breath_interpolated"])
    for k in ["insp", "exp"]
}

# precompute distance metrics for speed
# distances = {
#     breath_type: {metric: pdist(breath_mat, metric=metric) for metric in metrics}
#     for breath_type, breath_mat in datasets.items()
# }

umap_params = dict(
    breath_type=["insp"],  # , "exp"],
    n_neighbors=[5, 10, 100, 500],
    min_dist=[0.001, 0.01, 0.1, 0.5],
    metric=metrics,
)

# note: "breath_col" hardcoded to "breath_interpolated"

# make parameter combinations
conditions = []
for condition in product(*umap_params.values()):
    conditions.append({k: v for k, v in zip(umap_params.keys(), condition)})


# %%
# save data
# save all_breaths. drop interpolated; huge file
with open(save_folder.joinpath(f"all_breaths.pickle"), "wb") as f:
    pickle.dump(all_breaths, f)

with open(save_folder.joinpath("breaths_mats.pickle"), "wb") as f:
    pickle.dump(datasets, f)

# with open(save_folder.joinpath("distances.pickle"), "wb") as f:
#     pickle.dump(distances, f)

print(f"all_breaths saved to {save_folder}")

# %%
# make umap embeddings

errors = {}

# run gridsearch
for i, condition in enumerate(conditions):
    # insp or exp
    breath_type = condition.pop("breath_type")

    # reporting
    umap_name = f"embedding{i:03}-{breath_type}"

    # don't make new embedding if already extant
    if os.path.exists(save_folder.joinpath(f"{umap_name}.pickle")):
        print(f"#{i} exists! Skipping...")
        continue

    print(f"- Embedding {i:02} / {len(conditions):02} ({breath_type}):")
    print(f"\t- {condition}")

    # write to log
    with open(save_folder.joinpath(f"log.txt"), "a") as f:
        f.write(f"- embedding{i}:\n")

        for k,v in condition.items():
            f.write(f"  - {k}: {v}\n")

    # get data
    breaths_mat = datasets[breath_type]

    # # get distance
    # metric = condition.pop("metric")


    try:
        start_time = time.time()
        print("\t- Starting fit...")

        model = umap.UMAP(**condition, verbose=True)
        embedding = model.fit_transform(breaths_mat)
        
        print(f"\t- Done fitting! Took {time.time() - start_time}s.")
    except Exception as e:
        errors[i] = e
        print(f"\t- Error on #{i}! Skipping...")
        continue

    # plot umap
    n_since_stim = all_breaths.loc[all_breaths["type"] == breath_type, "trial_index"].fillna(-1)

    fig, ax = plt.subplots()

    sc = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=4,
        alpha=0.8,
        c=n_since_stim,
        cmap="viridis",
    )

    ax.set(
        xlabel="UMAP1",
        ylabel="UMAP2",
        title=umap_name,
    )

    cbar = fig.colorbar(sc, label="breaths since last stim")

    # save umap plot & embedding

    with open(save_folder.joinpath(f"{umap_name}.pickle"), "wb") as f:
        # formerly, saved model & embedding. with embeddings much bigger, this takes a lot of space. takes <1min to re-transform breath_mat
        pickle.dump(
            model,
            # {
            #     "model": model,
            #     "embedding": embedding,
            # },
            f,
        )

    fig.savefig(save_folder.joinpath(f"{umap_name}.jpg"))
    plt.close(fig)


# %%

fig, ax = plt.subplots()
# type = "insp"
# field = "amplitude"

# type, field = "insp", "amplitude"
# type, field = "insp", "duration_s"
breath_type, field = "exp", "duration_s"

bins = 100

# indices to separate
ii_type = all_breaths["type"] == breath_type
ii_call = all_breaths["putative_call"]

data = {
    "no_call": all_breaths.loc[ii_type & ~ii_call, field],
    "call": all_breaths.loc[ii_type & ii_call, field],
}

ax.hist(
    data.values(),
    bins=bins,
    label=data.keys(),
)
ax.set(
    xlabel=f"{breath_type} - {field}",
    ylabel="count",
    xlim=None,
)

ax.legend()

# %%
# look at short insps

short_insps = all_breaths.loc[
    (all_breaths["type"] == breath_type)
    & (~all_breaths["putative_call"])
    & (all_breaths["duration_s"] > 0.05)
]

fig, ax = plt.subplots()

ax.set_prop_cycle(plt.cycler("color", plt.cm.tab20.colors))

for i in np.linspace(0, len(short_insps) - 1, 20):
    insp = short_insps.iloc[int(i)]

    file, callnum = insp.name

    exp = all_breaths.loc[(file, callnum + 1)]

    x_start = 0
    for b in (insp, exp):
        breath = b["breath_norm"]
        x = (x_start + np.arange(len(breath))) / fs * 1000
        x_start += len(breath)
        ax.plot(x, breath)


# %%
# drop breaths which happened before any stimuli in block

# ii_pre_trial = all_breaths.index.to_frame()["stims_index"].isna()

# print(f"{sum(ii_pre_trial)} pre-trial breaths found. Dropping from all_breaths.")

# if len(ii_pre_trial) > 0:
#     pre_trial_breaths = all_breaths.loc[ii_pre_trial]

# all_breaths = all_breaths.loc[~ii_pre_trial]

# all_breaths
