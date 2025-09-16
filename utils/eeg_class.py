import numpy as np
from typing import List, Optional
# from utils.channel_class import EEG

class EEG:
    def __init__(self, data: np.ndarray, s_rate: float, montage: str, channels: Optional[List[str]] = None):
        """
        EEG data container.

        Parameters
        ----------
        data : np.ndarray
            EEG time-series, shape (n_channels, n_samples).
        sfreq : float
            Sampling frequency in Hz.
        channels : list of str, optional
            Channel names. Defaults to ["Ch0", "Ch1", ...].
        """
        self.data = np.asarray(data)
        self.s_rate = float(s_rate)
        self.n_channels, self.n_samples = self.data.shape
        self.channels = channels if channels is not None else [f"Ch{i}" for i in range(self.n_channels)]
        self.montage = montage

    @property
    def times(self) -> np.ndarray:
        """Return time vector in seconds."""
        return np.arange(self.n_samples) / self.s_rate

    @property
    def duration(self) -> float:
        """Return total duration of the recording in seconds."""
        return self.n_samples / self.s_rate

    def get_channel(self, ch: str) -> np.ndarray:
        """Return data for a single channel by name."""
        if ch not in self.channels:
            raise ValueError(f"Channel '{ch}' not found.")
        idx = self.channels.index(ch)
        return self.data[idx]

    def bandpass(self, low: float, high: float):
        """Example: placeholder for filtering."""
        # You can plug in scipy.signal here
        raise NotImplementedError("Add filtering with scipy.signal.filtfilt")

    def __repr__(self):
        return (f"<EEG | {self.n_channels} channels, {self.n_samples} samples, "
                f"sfreq={self.s_rate} Hz, duration={self.duration:.2f}s>")
