# ./utils/video.py
# 2024.05.13 CDR
#
# Functions related to processing callback experiment videos
#


def get_video_frames_from_callback_audio(
    camera_channel_audio: np.ndarray,
    threshold_function=lambda x: np.max(x) * 0.2,
    do_check=True,
) -> np.ndarray:
    """
    - takes ONE CHANNEL of audio as a numpy array
    - return all frames when audio starts dips below threshold (ie, frame i for i<=thresh iff (i-1)>thresh)
    - if do_check is True, ensures that range of inter-frame intervals <= 10
    """
    import numpy as np

    threshold = threshold_function(camera_channel_audio)

    a_thresholded = camera_channel_audio <= threshold  # to bool

    # one frame prior. don't allow first frame
    offset = np.append([1], a_thresholded[:-1])

    frames = np.nonzero(a_thresholded & ~offset)[0]

    if do_check:
        allowable_range = 10

        # number of audio samples between subsequent frames
        deltas = frames[1:] - frames[:-1]

        assert (
            np.ptp(deltas) <= allowable_range
        ), "Warning! Frame timings might vary too much. You might need a stricter threshold function."

    return frames
