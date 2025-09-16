import mne
from mne.time_frequency import tfr_array_morlet
import yaml
import numpy as np
from prettytable import PrettyTable, SINGLE_BORDER  # pip install prettytable
from utils.constants import BASELINE_DURATION, CANONICAL_BANDS
import matplotlib.pyplot as plt
import os
mne.set_log_level("ERROR")


def configure(config_path: str, title: str = "Configuration Settings"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Extract parameters
    pre_sham_path  = cfg["pre_sham_path"]
    post_sham_path  = cfg["post_sham_path"]
    pre_active_path = cfg["pre_active_path"]
    post_active_path = cfg["post_active_path"]
    montage        = cfg.get("montage")
    window         = float(cfg.get("window_length", 2.0))
    overlap        = float(cfg.get("overlap", 0.0))
    channels       = cfg.get("channels")

    # Build table
    table = PrettyTable()
    table.set_style(SINGLE_BORDER)        # optional: nice single-line borders
    table.title = title                    # <-- boxed, spanning title
    table.field_names = ["Setting", "Value"]
    table.align["Setting"] = "r"
    table.align["Value"]   = "l"
    # table.max_width = 100                # optional: wrap long values

    table.add_row(["pre_sham_path",  pre_sham_path])
    table.add_row(["post_sham_path", post_sham_path])
    table.add_row(["pre_active_path", pre_active_path])
    table.add_row(["post_active_path", post_active_path])
    table.add_row(["montage",        montage])
    table.add_row(["window",         window])
    table.add_row(["overlap",        overlap])
    table.add_row(["channels", channels])

    print(table)
    return pre_sham_path, post_sham_path, pre_active_path, post_active_path, montage, window, overlap, channels



def load_pre_post_stim(edf_file: str, channels, montage: str):
    # Load the EDF files into MNE Raw objects
    mne_data = mne.io.read_raw_edf(edf_file, preload=True)
    mne_data.pick(channels)
    mne_data.set_montage(montage, match_case=False)
    return mne_data

def cmor_tfr(raw_csd,
                      freqs=np.linspace(1, 45, 64),
                      n_cycles=6,
                      output="power",
                      n_jobs=-1,         # <- int or None
                      decim=1):
    assert isinstance(raw_csd, mne.io.BaseRaw), "Pass a Raw (CSD) object"
    X = raw_csd.get_data()                 # (n_channels, n_times)
    sf = float(raw_csd.info["sfreq"])
    assert X.shape[0] > 0 and X.shape[1] > 0, "No channels or samples"

    # Shape for tfr_array_morlet: (n_epochs, n_channels, n_times)
    X_epo = X[None, ...]

    # Coerce n_jobs in case someone passes "auto"
    if isinstance(n_jobs, str):
        n_jobs = os.cpu_count()  # or set to None

    power = tfr_array_morlet(
        X_epo, sfreq=sf, freqs=freqs, n_cycles=n_cycles,
        output=output, use_fft=True, n_jobs=n_jobs, decim=decim
    )  # (1, n_channels, n_freqs, n_times/decim)

    return power[0], freqs

def baseline_correct(raw: mne.io.Raw) -> mne.io.Raw:
    """Vectorized: subtract mean of the first BASELINE_DURATION seconds from each channel."""
    sfreq = raw.info["sfreq"]
    n_samples = int(BASELINE_DURATION * sfreq)

    # Get full data matrix (n_channels, n_times)
    data = raw.get_data()

    # Compute mean of the first n_samples per channel
    n0 = min(n_samples, data.shape[1])  # handle short recordings safely
    baseline = data[:, :n0].mean(axis=1, keepdims=True)

    # Subtract baseline from all samples in one go
    raw._data -= baseline
    return raw

def epoch_raw(raw: mne.io.Raw, window_sec: float, overlap_sec: float,
              picks="eeg", preload=True, decim=1) -> mne.Epochs:
    sfreq = raw.info["sfreq"]
    duration = window_sec
    step = max(1e-12, window_sec - overlap_sec)  # seconds, avoid zero/negative
    events = mne.make_fixed_length_events(raw, duration=step, start=0.0)
    # `Epochs` picks the time span around each event = `tmin=0, tmax=duration`
    epochs = mne.Epochs(
        raw, events=events, tmin=0.0, tmax=duration, picks=picks,
        preload=preload, decim=decim, baseline=None, reject_by_annotation=True,
        verbose="ERROR"
    )
    return epochs

def plot_scalogram(power, freqs, times, ch_idx, ch_name=None, vmin=None, vmax=None):
    """
    Plot a time-frequency scalogram for one channel.

    Parameters
    ----------
    power : ndarray, shape (n_channels, n_freqs, n_times)
        TFR power (linear or dB).
    freqs : ndarray
        Frequencies corresponding to axis 1.
    times : ndarray
        Time vector in seconds, length = n_times.
    ch_idx : int
        Channel index to plot.
    ch_name : str, optional
        Channel label for the title.
    vmin, vmax : float, optional
        Color scaling.
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(
        power[ch_idx],
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    plt.colorbar(label="Power")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    title = f"Scalogram — channel {ch_name if ch_name else ch_idx}"
    plt.title(title)
    plt.tight_layout()
    plt.show()