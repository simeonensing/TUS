import mne
from utils.cmor import decompose_epochs_vectorized
from utils.helpers import load_pre_post_stim, configure, baseline_correct, epoch_raw
from utils.constants import CANONICAL_BANDS
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence


def _flatten_series(band_ECBT: np.ndarray, epochs: mne.Epochs,
                    window_sec: float, chan_idx: int, band_idx: int):
    """Return (t, y) flattened across epochs for one (channel, band)."""
    E, C, B, T = band_ECBT.shape
    y = band_ECBT[:, chan_idx, band_idx, :].reshape(-1)  # (E*T,)
    sfreq = float(epochs.info["sfreq"])
    starts = epochs.events[:, 0] / sfreq                 # (E,) seconds in raw timeline
    starts -= starts.min()                               # start at 0 s for this recording
    t_rel = np.linspace(0, window_sec, T, endpoint=False)
    t = (starts[:, None] + t_rel[None, :]).reshape(-1)   # (E*T,)
    return t, y

def plot_all_channels_bands_grid(
    band_pre: np.ndarray, band_post: np.ndarray,   # (E, C, B, T)
    epochs_pre: mne.Epochs, epochs_post: mne.Epochs,
    window_sec: float, band_names: Sequence[str],
    channels: Sequence[str] | None = None,
    decim_plot: int = 1,                           # downsample for faster plotting
    unit_label: str = "band power (a.u.)",
):
    """
    Tiled grid: rows=Channels, cols=Bands. Each cell overlays Pre vs Post time-courses.
    """
    # ch_names = list(channels) if channels else epochs_pre.ch_names
    ch_names = ['F8']
    # sanity: make sure channels exist in both objects
    idx_pre  = [epochs_pre.ch_names.index(ch)  for ch in ch_names]
    idx_post = [epochs_post.ch_names.index(ch) for ch in ch_names]

    n_ch = len(ch_names)
    band_names = ['gamma']
    n_b  = len(band_names)

    # figure sizes: ~3 in per column, ~1.4 in per row
    fig, axes = plt.subplots(
        nrows=n_ch, ncols=n_b, figsize=(3*n_b, max(1.4*n_ch, 5)),
        sharex='col', sharey='row', squeeze=False
    )

    for r, (ch, ci_pre, ci_post) in enumerate(zip(ch_names, idx_pre, idx_post)):
        for c, band in enumerate(band_names):
            ax = axes[r, c]
            # build flattened series for this (channel, band)
            t_pre,  y_pre  = _flatten_series(band_pre,  epochs_pre,  window_sec, ci_pre,  c)
            t_post, y_post = _flatten_series(band_post, epochs_post, window_sec, ci_post, c)

            if decim_plot > 1:
                t_pre,  y_pre  = t_pre[::decim_plot],  y_pre[::decim_plot]
                t_post, y_post = t_post[::decim_plot], y_post[::decim_plot]

            ax.plot(t_pre,  y_pre,  lw=0.8, label="Pre"  if (r==0 and c==0) else None)
            ax.plot(t_post, y_post, lw=0.8, label="Post" if (r==0 and c==0) else None)

            if r == 0:
                ax.set_title(band, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{ch}\n{unit_label}", fontsize=9)
            if r == n_ch - 1:
                ax.set_xlabel("Time (s)")

            # ax.hist(y_pre,  bins=30, alpha=0.5, label="Pre"  if (r==0 and c==0) else None)
            # ax.hist(y_post, bins=30, alpha=0.5, label="Post" if (r==0 and c==0) else None)
            # if r == 0:
            #     ax.set_title(band, fontsize=10)
            # if c == 0:
            #     ax.set_ylabel("Count", fontsize=9)
            # if r == n_ch - 1:
            #     ax.set_xlabel("Relative Power [dB]")

    # one legend for the whole figure
    handles, labels = axes[0,0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()

if __name__ == "__main__":
    # Define edf paths
    pre_stim_path, post_stim_path, montage, window_dur, overlap, channels = configure("config.yaml")
    # Read edfs into
    raw_pre,raw_post = load_pre_post_stim(pre_stim_path,post_stim_path)
    # Baseline correct
    baseline_pre = baseline_correct(raw_pre)
    baseline_post = baseline_correct(raw_post)
    epochs_pre = epoch_raw(baseline_pre, window_dur, overlap,
                  picks=channels, preload=True, decim=1)
    epochs_post = epoch_raw(baseline_post, window_dur, overlap,
                           picks=channels, preload=True, decim=1)
    # Pre/Post epochs you already created
    band_pre, names, freqs = decompose_epochs_vectorized(
        epochs_pre, CANONICAL_BANDS, decim=2, output="power", average=None
    )  # (E, C, B, T)

    band_post, _, _ = decompose_epochs_vectorized(
        epochs_post, CANONICAL_BANDS, decim=2, output="power", average=None
    )

    # After you computed band_pre / band_post with shape (E, C, B, T)
    plot_all_channels_bands_grid(
        band_pre=band_pre,
        band_post=band_post,
        epochs_pre=epochs_pre,
        epochs_post=epochs_post,
        window_sec=window_dur,
        band_names=names,  # returned by decompose_epochs_vectorized
        channels=channels,  # from your YAML; or None to use all
        decim_plot=5,  # tweak for speed/clarity
        unit_label="dB"  # if you normalized; else "a.u."
    )

