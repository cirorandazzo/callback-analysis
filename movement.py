import os

import cv2 as cv
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy.io import wavfile

from utils.audio import AudioObject
from utils.video import (
    get_triggers_from_audio,
    get_video_frames_from_callback_audio,
    play_video,
    write_diff_video,
)


def get_loom_movements(window_start, window_length, movement_ii, fs):
    import numpy as np

    window_end = window_start + window_length

    movement_times = movement_ii[
        (movement_ii > window_start) & (movement_ii <= window_end)
    ]
    movement_times = (movement_times - window_start) * 1000 / fs

    if not any(vid_frames_ii >= window_end):
        movement_count = np.NaN
    else:  # ensure data exists for the whole window
        movement_count = len(movement_times)

    latency_ms = min(movement_times, default=np.NaN)

    return latency_ms, movement_count, movement_times


def get_loom_latency_df(loom_onsets, window_length, movement_ii, fs):
    import pandas as pd

    df = pd.DataFrame()

    df["loom_onset"] = loom_onsets

    out = df["loom_onset"].apply(
        get_loom_movements,
        args=(window_length, movement_ii, fs),
    )

    out = pd.DataFrame(
        [[l, c, t] for l, c, t in out.values],
        columns=["Latency", "Count", "Times"],
    )

    df = pd.concat([df, out], axis=1)
    return df


def get_max_avg_pixel_changes(diff_video_fname):
    import numpy as np

    from utils.video import open_video

    max_change = [0]
    avg_change = [0]

    capture = open_video(diff_video_fname)

    while True:
        ret, frame = capture.read()

        if frame is None:
            break
        else:
            max_change.append(np.max(frame))
            avg_change.append(np.mean(frame))

    max_change = np.array(max_change)
    avg_change = np.array(avg_change)

    return (max_change, avg_change)


def plot_latency_by_trial(
    data,
    color_data=[0, 0, 0],
    color_blocks=[0.7, 0.7, 0.7],
    ax=None,
):

    if ax is None:
        fig, ax = plt.subplots()

    latency = data.reset_index()["Latency"]

    # plot line
    ax.plot(latency, color=color_data, marker="o")

    nr = latency[np.isnan(latency)].index
    ax.scatter(
        nr,
        np.zeros_like(nr),
        marker="x",
        color=color_data,
        label="n.r.",
    )

    ax.set(
        xlabel="Trial",
        ylabel="Movement latency"
    )
    ax.legend()

    blocks = data.reset_index()["block"]
    for bl in np.unique(blocks):
        this_bl = blocks[blocks == bl].index

        if int(bl) % 2 == 1:
            continue
        ax.axvspan(
            this_bl.min() - 0.5,
            this_bl.max() + 0.5,
            facecolor=color_blocks,
            alpha=0.2,
        )

    new_blocks = data.reset_index()[data.reset_index()["loom"] == 0]
    sec_ax = ax.secondary_xaxis(location="top")
    sec_ax.set(
        xlabel="Block",
        xticks=new_blocks.index - 0.5,
        xticklabels=new_blocks["block"],
    )

    return ax


if __name__ == "__main__":
    verbose = True
    do_plot = True

    birdname = "or54rd45"
    # birdname = "or14pu27"
    pattern = f"/Volumes/users/randazzo/callbacks/loom_only/{birdname}/loom_only/*_CAM0-0000.avi"
    cam_n = 1
    diff_video_folder = "/Volumes/PlasticBag/anxiety_calls/loom_only-diff_videos"

    fs = 44100

    ignore_range_fr = round(fs * 0.2)

    movement_threshold_function = lambda x: np.median(x) * 1.2
    suppress_remake_diff = True
    window_length = round(300 * (fs / 1000))  # 300 ms --> fr

    os.makedirs(diff_video_folder, exist_ok=True)

    all_data = []

    paths = [f.split("_CAM0-0000.avi")[0] for f in glob.glob(pattern)]
    # w/o extension or camera info; should match for wav/avi
    for rootname in paths:
        path, fname = os.path.split(rootname)

        if verbose:
            print(f"Starting file:\t{fname}")

        # %% get paths
        wav_path = rootname + ".wav"
        avi_path = rootname + f"_CAM{cam_n}-0000.avi"
        diff_video_fname = os.path.join(
            diff_video_folder, fname + f"_CAM{cam_n}-DIFF.avi"
        )

        # %% load audio & separate useful channels
        a_fs, audio = wavfile.read(wav_path)

        assert fs == a_fs

        bird_audio = AudioObject.from_wav(wav_path, channel=0)
        bird_audio.filtfilt_butter_default()
        bird_audio.rectify_smooth(smooth_window_f=round(2 * 44.1))

        cam_frames = audio[:, 1]
        loom_trigs = audio[:, 4]

        # %% get loom & video frame timestamps
        loom_onsets = get_triggers_from_audio(loom_trigs)
        vid_frames_ii = get_video_frames_from_callback_audio(
            cam_frames,
            threshold_function=lambda x: np.max(x) * 0.3,
            ignore_range_fr = ignore_range_fr,
        )

        # %% make diff video
        if not (suppress_remake_diff and os.path.exists(diff_video_fname)):
            write_diff_video(avi_path, diff_video_fname)

            if verbose:
                print(f"Wrote diff video: {diff_video_fname}")

        elif verbose:
            print("suppress_remake_diff is on & diff video already exists! Skipping...")

        # %% track movement
        max_change, avg_change = get_max_avg_pixel_changes(diff_video_fname)

        # only consider video frames during audio
        max_change = max_change[: len(vid_frames_ii)]
        avg_change = avg_change[: len(vid_frames_ii)]

        # get movement timings by thresholding
        thresholded_movement_ii_vid = get_triggers_from_audio(
            avg_change,
            threshold_function=movement_threshold_function,
            crossing_direction="up",
        )

        # convert indices to audio timing (frames)
        movement_ii = vid_frames_ii[thresholded_movement_ii_vid]

        # %% get movement information for each loom trigger
        this_df = get_loom_latency_df(loom_onsets, window_length, movement_ii, fs)
        this_df.index.name = "loom"
        this_df["birdname"] = birdname
        this_df["filename"] = avi_path
        this_df["block"] = rootname.split("Block")[1]
        this_df.set_index(["birdname", "block"], append=True, inplace=True)

        all_data.append(this_df)

    # %% concat & save
    all_data = pd.concat(all_data)
    # all_data.reorder_levels(["birdname", "block", "loom"])
    all_data.index = all_data.index.reorder_levels(["birdname", "block", "loom"])
    all_data.sort_index(inplace=True)

    if do_plot:
        ax = plot_latency_by_trial(all_data)
        ax.set(title=birdname)

        plt.savefig(
            os.path.join(diff_video_folder, f"mvmt_latency-{birdname}.png"),
            dpi=400
        )

    with open(
        os.path.join(diff_video_folder, f"all_data-{birdname}.pickle"), "wb"
    ) as f:
        pickle.dump(all_data, f)

    if verbose:
        print(f"Finished! Saved dataframe to:\t{diff_video_folder}")
