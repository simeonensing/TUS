import mne
import yaml
import numpy as np
from prettytable import PrettyTable, SINGLE_BORDER  # pip install prettytable
from utils.constants import BASELINE_DURATION, CANONICAL_BANDS
from mne.time_frequency import tfr_array_morlet

mne.set_log_level("ERROR")


def configure(config_path: str, title: str = "Configuration Settings"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Extract parameters
    pre_stim_path  = cfg["pre_stim_path"]
    post_stim_path = cfg["post_stim_path"]
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

    table.add_row(["pre_stim_path",  pre_stim_path])
    table.add_row(["post_stim_path", post_stim_path])
    table.add_row(["montage",        montage])
    table.add_row(["window",         window])
    table.add_row(["overlap",        overlap])
    table.add_row(["channels", channels])

    print(table)
    return pre_stim_path, post_stim_path, montage, window, overlap, channels



def load_pre_post_stim(pre_file: str, post_file: str):
    # Load the EDF files into MNE Raw objects
    raw_pre = mne.io.read_raw_edf(pre_file, preload=True)
    raw_post = mne.io.read_raw_edf(post_file, preload=True)

    return raw_pre, raw_post

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