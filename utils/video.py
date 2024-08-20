# ./utils/video.py
# 2024.05.13 CDR
#
# Functions related to processing callback experiment videos
#


def get_triggers_from_audio(
    audio: np.ndarray,
    threshold_function=lambda x: 10 * np.mean(x),
    crossing_direction="up",
    allowable_range=None,
) -> np.ndarray:
    import numpy as np

    threshold = threshold_function(audio)

    if crossing_direction == "down":  # get downward crossings; eg, video frames
        a_thresholded = audio <= threshold
    elif crossing_direction == "up":  # get upward crossings
        a_thresholded = audio >= threshold
    else:
        raise ValueError(
            f"'{crossing_direction}' is not a valid crossing_direction. Must be 'up' or 'down'."
        )

    # one frame prior. don't allow first frame
    offset = np.append([1], a_thresholded[:-1])

    frames = np.nonzero(a_thresholded & ~offset)[0]

    if allowable_range is not None:
        # number of audio samples between subsequent frames
        deltas = frames[1:] - frames[:-1]

        check = False
        r = np.ptp(deltas)

        if np.isscalar(allowable_range):  # just one number
            check = r <= allowable_range
        else:
            check = (r >= allowable_range[0] & r <= allowable_range[1]).all()

        assert (
            check
        ), "Warning! Frame timings might vary too much. You might need a stricter threshold function."

    return frames


def get_video_frames_from_callback_audio(
    camera_channel_audio: np.ndarray,
    threshold_function=lambda x: np.max(x) * 0.2,
    allowable_range=10,
    **kwargs,
) -> np.ndarray:
    """
    - takes ONE CHANNEL of audio as a numpy array
    - return all frames when audio starts dips below threshold (ie, frame i for i<=thresh iff (i-1)>thresh)
    - if do_check is True, ensures that range of inter-frame intervals <= 10
    """
    frames = get_triggers_from_audio(
        audio=camera_channel_audio,
        threshold_function=threshold_function,
        crossing_direction="down",
        allowable_range=10,
        **kwargs,
    )

    return frames
