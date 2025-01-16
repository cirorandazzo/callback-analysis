# %%
#
# %load_ext autoreload
# %autoreload 1

import glob
import numpy as np
import pandas as pd

import os


# %aimport utils.callbacks
from utils.callbacks import call_mat_stim_trial_loader
from utils.file import multi_index_from_dict, parse_parameter_from_string

# %%
# OPTIONS

bird = "pk74br76"
processed_directory = (
    rf"M:\public\callback experiments\HVC_pharmacology\pk74br76\**\*.not.mat"
)

save_csv_directory = "./data/hvc_pharm"
save_pickle_path = os.path.join(save_csv_directory, f"{bird}.pickle")

# any stimulus_trials containing call types NOT in this list are excluded (this includes unlabeled, which are stored as 'USV'!!)
acceptable_call_labels = ["Call", "Stimulus"]

files = [f for f in glob.glob(processed_directory) if not "PostBlock" in f]

files

# %%
# DEFINE PARAMETER PARSING


def get_params_from_pharmacology(filepath):

    file = os.path.split(filepath)[-1]

    params = file.split("-")

    assert (
        len(params) == 6
    ), f"Looks like filename might be formatted wrong! File: {filepath}"

    birdname, day, drug, time, datetime, block = params

    # good as is: birdname, drug
    # ignore datetime for now - leading zeros lost

    # day: remove leading "d", make int
    day = int(day[1:])

    # time: replace _ --> ., remove trailing "h", make float
    time = float(time.replace("_", ".")[:-1])

    block = block.replace(".wav.not.mat", "")
    block = int(parse_parameter_from_string(block, "Block", chars_to_ignore=0))

    params = dict(
        birdname=birdname,
        day=day,
        drug=drug,
        time=time,
        block=block,
    )

    return params


# %%
# MAKE DF

df = pd.DataFrame()

call_types_all = pd.DataFrame()
rejected_trials_all = pd.DataFrame()
calls_all = pd.DataFrame()

for file in files:

    try:
        calls_df, stim_trials, rejected_trials, file_info, call_types = (
            call_mat_stim_trial_loader(
                file,
                acceptable_call_labels=acceptable_call_labels,
                from_notmat=True,
                verbose=False,
            )
        )
    except TypeError:
        print(f"Failed to make dataframe for file: {file}")
        continue

    params = get_params_from_pharmacology(file)

    # create multiindex: birdname, stim_trial_index, call_index
    stim_trials = multi_index_from_dict(stim_trials, params, keep_current_index=True)
    df = pd.concat((df, stim_trials), axis="rows")

    rejected_trials = multi_index_from_dict(
        rejected_trials, params, keep_current_index=True
    )
    rejected_trials_all = pd.concat((rejected_trials_all, rejected_trials), axis="rows")

    call_types = multi_index_from_dict(call_types, params, keep_current_index=True)
    call_types_all = pd.concat((call_types_all, call_types), axis="rows")

    calls_df = multi_index_from_dict(calls_df, params, keep_current_index=True)
    calls_all = pd.concat((calls_all, calls_df), axis="rows")

df.sort_index(inplace=True)

print("df:")
df

# %%
# REPORT REJECTED TRIALS

print("Rejected trials:")
rejected_trials_all

# %%
print(
    "Call types in rejected trials."
    + "\nLabel `USV` means an accepted call was not given a label."
    + "\nGo back to DeepSqueak & fix ths."
)

rejected_trial_call_types = call_types_all.loc[rejected_trials_all.index]
rejected_trial_call_types
# TODO: add stim index to rej trial type df (is this the first stim?)

# # see only blocks with a specific call type
#
# label = 'USV'
# label = 'Noise'
# call_types_all.loc[~np.isnan(call_types_all.loc[:, label])]

# %%
all_birds = list(set(df.index.get_level_values(0)))
all_birds

# %%
# eliminate all block 0s - account for first loom bug

# raise Exception('Make sure you want to do this! You will need to reload the data afterward if you want block 0 back.')

# blocks = df.index.get_level_values(2)
# df = df[blocks != 0]

df

# %%
# plotting conveniences

level_names = df.index.names
df.reset_index(inplace=True)

# add suffix "_washout"
df["drug"] = df.apply(
    lambda row: row["drug"] + "_washout" if row["time"] >= 6.5 else row["drug"], axis=1
)

# make washout day n + 0.5
df["day"] = df.apply(
    lambda row: row["day"] + 0.5 if row["time"] >= 6.5 else row["day"], axis=1
)

# "raster_timepoint" for correct ordering

df["raster_timepoint"] = df.apply(lambda row: row["time"] + 0.01 * row["block"], axis=1)

df.set_index(level_names, inplace=True)

df

# %%
if save_pickle_path is not None:
    import pickle

    to_save = dict(
        all_birds=all_birds,
        df=df,
        rejected_trials_all=rejected_trials_all,
        calls_all=calls_all,
        call_types_all=call_types_all,
    )

    with open(save_pickle_path, "wb") as f:
        pickle.dump(to_save, file=f)

print(f"Saved to: {save_pickle_path}")

# %%
if save_csv_directory is not None:
    df.to_csv(os.path.join(save_csv_directory, f"{bird}-trials.csv"))
    calls_all.to_csv(os.path.join(save_csv_directory, f"{bird}-calls.csv"))

print(f"Saved to: {save_csv_directory}")
