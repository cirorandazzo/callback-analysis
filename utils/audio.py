import numpy as np


class AudioObject:
    def __init__(
        self,
        audio,
        fs,
        b=None,
        a=None,
        do_spectrogram=False,
    ):
        """
        b,a: Numerator (b) and denominator (a) polynomials of the IIR filter
        """
        import numpy as np

        self.audio = audio
        self.fs = fs
        self.audio_frs = None

        if b is not None and a is not None:
            self.filtfilt(b, a)
        else:
            self.audio_filt = None

        if do_spectrogram and self.audio_filt is not None:
            self.make_spectrogram()
        else:
            self.spectrogram = None

    @classmethod
    def from_wav(
        cls,
        filename,
        channel=0,
        **kwargs,
    ):
        """
        Create an AudioObject given a wav file & IIR filter details.

        Args should match default constructor.
        """
        from scipy.io import wavfile

        fs, audio = wavfile.read(filename)
        audio = audio[:, channel]

        new_obj = cls(
            audio,
            fs,
            **kwargs,
        )

        new_obj.file = filename

        return new_obj

    @classmethod
    def from_cbin(
        cls,
        filename,
        channel=0,
        **kwargs,
    ):
        """
        Create an AudioObject given a cbin file. Defaults to first channel (0) which can be changed as a keyword arg.

        Args should match default constructor.
        """
        from utils.evfuncs import load_cbin

        audio, fs = load_cbin(filename, channel=channel)

        new_obj = cls(
            audio,
            fs,
            **kwargs,
        )

        new_obj.file = filename

        return new_obj

    def filtfilt(self, b, a):
        from scipy.signal import filtfilt

        self.audio_filt = filtfilt(b, a, self.audio)

    def filtfilt_butter_default(self, f_low=500, f_high=15000, poles=8):
        from scipy.signal import butter

        b, a = butter(poles, [f_low, f_high], btype="bandpass", fs=self.fs)

        self.filtfilt(b, a)

    def rectify_smooth(self, smooth_window_f):
        import numpy as np

        if self.audio_filt is None:
            raise UnfilteredException()

        rectified = np.power(self.audio_filt, 2)

        # cumsum = np.cumsum(np.insert(rectified, 0, 0))
        # smoothed = (cumsum[smooth_window_f:] - cumsum[:-smooth_window_f]) / float(smooth_window_f)

        wind = np.ones(smooth_window_f)
        smoothed = np.convolve(wind, rectified, "same")

        self.audio_frs = smoothed

    def make_spectrogram(
        self,
        n=1024,
        overlap=1020,
        normalize_range=(0, 1),
    ):
        from scipy.signal.windows import hamming
        from scipy.signal import ShortTimeFFT

        if self.audio_filt is None:
            raise UnfilteredException()

        window = hamming(n)
        hop = n - overlap

        self.SFT = ShortTimeFFT(
            window,
            hop,
            self.fs,
            fft_mode="onesided",
        )

        spx = self.SFT.spectrogram(self.audio_filt)
        spx = np.log10(spx)

        if normalize_range is not None:
            spx = normalize(spx, normalize_range)

        self.spectrogram = spx

    def plot_spectrogram(self, **kwargs):
        if self.spectrogram is None:
            self.make_spectrogram()

        return plot_spectrogram(self.spectrogram, self.SFT, **kwargs)

    def get_sample_times(self):
        """
        Return an array of sample times; eg, to use when plotting.
        """
        return np.arange(len(self.audio)) / self.fs

    def get_length_s(self):
        """
        Return length of audio in s.
        """

        return len(self.audio) / self.fs


def plot_spectrogram(
    spectrogram: np.ndarray,
    SFT: np.ndarray,
    ax=None,
    x_offset_s=0,
    **plot_kwargs,
):
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots()

    # extent: times of audio signal (s) & frequencies (Hz). for correct axis labels
    extent = np.array(SFT.extent(SFT.hop * spectrogram.shape[1])).astype("float")
    extent[0:2] += x_offset_s  # offset x axis

    if "cmap" not in plot_kwargs.keys():
        plot_kwargs["cmap"] = "bone_r"

    ax.imshow(
        spectrogram,
        origin="lower",
        aspect="auto",
        extent=extent,
        **plot_kwargs,
    )

    ax.set(
        xlabel="Time (s)",
        ylabel="Frequency (Hz)",
    )

    return ax


def normalize(x, range=(-1, 1)):
    from sklearn.preprocessing import minmax_scale

    flattened = minmax_scale(x.flatten(), feature_range=range).astype("float32")
    return flattened.reshape(x.shape)


class UnfilteredException(Exception):
    def __init__(self):
        self.message = "Filter audio with self.filtfilt before calling this method!"


if __name__ == "__main__()":
    # audio obj example usage
    from scipy.io import wavfile
    from scipy.signal import butter

    # load audio
    wav_path = "/Users/cirorandazzo/code/callback-analysis/tests/0toes2mics20240801122415-Stim0-Block0.wav"
    fs, audio = wavfile.read(wav_path)
    audio = audio[:, 0]  # only take first channel

    # Filter parameters: 8pole butterpass bandworth filter or wtv the kids are saying these days
    f_low, f_high = (500, 15000)
    b, a = butter(8, [f_low, f_high], btype="bandpass", fs=fs)

    n = 1024  # window length
    overlap = 1020

    plot_spectrogram_kwargs = {
        "cmap": "magma",
        "plot_kwargs": {
            "ylim": (0, 15000),
        },
    }

    bird_audio = AudioObject(
        normalize(audio[:, 0]),
        fs,
        b,
        a,
        # do_spectrogram=True,  # plots spectrogram with default params
    )

    bird_audio.make_spectrogram(
        n,
        overlap,
        normalize_range=(0, 1),
    )

    ax = bird_audio.plot_spectrogram()
