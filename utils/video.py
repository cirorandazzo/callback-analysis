# ./utils/video.py
# 2024.05.13 CDR
#
# Functions related to processing callback experiment videos
#


def get_video_frames_from_callback_audio(camera_channel_audio):
    """
    takes ONE CHANNEL of audio as a numpy array, and returns indices when that crosses mean value (threshold)
    """
    import numpy as np

    threshold = np.mean(camera_channel_audio)

    a_thresholded = camera_channel_audio > threshold  # to bool
    offset = np.append([0], a_thresholded[:-1])

    # return a_thresholded & ~offset
    return np.nonzero(a_thresholded & ~offset)[0]
