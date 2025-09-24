#!/usr/bin/env python3
"""
Topomaps of CSD band-power change using a COMMON BASELINE (pre-active)
for canonical EEG bands:
  δ (1–4 Hz), θ (4–8 Hz), α (8–12 Hz), β (12–30 Hz), γ (30–45 Hz)

Per band we plot two maps:
- Left  ("Before TUS"):  post-sham   vs pre-active  → Δ{band} Power [dB]
- Right ("After TUS"):   post-active vs pre-active  → Δ{band} Power [dB]

Why: using the same baseline avoids bias from a mismatched pre-sham baseline.
"""

from pathlib import Path
import numpy as np
import mne
from mne.preprocessing import compute_current_source_density
from scipy.signal import hilbert
import matplotlib
import matplotlib.pyplot as plt

# -------------------- YOUR FOUR EDF PATHS --------------------
pre_sham_path    = Path("data/TUS/EEG 1_EPOCX_218863_2025.08.28T10.05.54+12.00.edf")
post_sham_path   = Path("data/TUS/EEG 2_EPOCX_218863_2025.08.28T10.34.10+12.00.edf")
pre_active_path  = Path("data/TUS/EEG 3_EPOCX_218863_2025.08.28T10.58.26+12.00.edf")  # COMMON BASELINE
post_active_path = Path("data/TUS/EEG 4_EPOCX_218863_2025.08.28T11.31.55+12.00.edf")
# -------------------------------------------------------------

# Your 14 EEG channels (10–10 / 1005)
CHANNELS_1005 = [
    "AF3","F7","F3","FC5","T7","P7","O1","O2",
    "P8","T8","FC6","F4","F8","AF4"
]

# ---------- Processing parameters ----------
montage_name   = "standard_1005"
resample_sfreq = 250               # None = keep native sampling rate
smooth_sec     = 0.0               # optional moving-average (s) on per-channel power
save_figs      = False             # set True to save PNGs
save_dir       = Path("figs_topomaps")  # used only if save_figs=True
# Canonical bands: label -> (low, high)
BANDS = {
    # "δ": (1.0, 4.0),
    # "θ": (4.0, 8.0),
    # "α": (8.0, 12.0),
    # "β": (12.0, 30.0),
    "γ": (30.0, 45.0),
}
# ------------------------------------------

# Safer backend for headless environments
try:
    if "DISPLAY" not in __import__("os").environ:
        matplotlib.use("Agg")
except Exception:
    pass


# ------------------------- HELPERS ---------------------------

def _drop_bad_position_channels(raw: mne.io.BaseRaw) -> list[str]:
    bad = []
    for ch in raw.info["chs"]:
        if ch["kind"] != mne.io.constants.FIFF.FIFFV_EEG_CH:
            continue
        xyz = ch["loc"][:3]
        if not np.isfinite(xyz).all() or np.allclose(xyz, 0.0):
            bad.append(ch["ch_name"])
    if bad:
        raw.drop_channels(bad)
        print(f"[CSD] Dropped {len(bad)} EEG channel(s) without valid positions: {bad}")
    return bad


def load_raw_edf(path: Path, montage: str | None, sfreq: float | None) -> mne.io.BaseRaw:
    if not path.exists():
        raise FileNotFoundError(f"EDF not found: {path}")
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.pick(picks="eeg")
    if montage:
        raw.set_montage(montage, match_case=False, on_missing="ignore")
    mne.set_eeg_reference(raw, "average", projection=False, verbose="ERROR")
    have = [ch for ch in CHANNELS_1005 if ch in raw.ch_names]
    missing = [ch for ch in CHANNELS_1005 if ch not in raw.ch_names]
    if missing:
        print(f"[warn] Missing in {path.name}: {missing}")
    if have:
        raw.pick(have)
    if sfreq:
        raw.resample(float(sfreq))
    _drop_bad_position_channels(raw)
    if len(mne.pick_types(raw.info, eeg=True)) < 4:
        raise RuntimeError(f"Too few EEG channels with positions after cleanup in {path.name}. Present: {raw.ch_names}")
    return raw


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    x_pad = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(k) / k
    return np.convolve(x_pad, kernel, mode="valid")


def csd_band_power_per_channel(raw: mne.io.BaseRaw, band=(30.0, 45.0)) -> tuple[list[str], np.ndarray, mne.Info]:
    """CSD -> bandpass(band) -> Hilbert power -> time-mean per channel."""
    if raw.get_montage() is None:
        raise RuntimeError("CSD requires a montage with channel positions.")
    events = np.array([[int(raw.times[0]), 0, 1]])
    epochs = mne.Epochs(raw, events, event_id={"all": 1},
                        tmin=raw.times[0], tmax=raw.times[-1],
                        baseline=None, preload=True, verbose="ERROR")
    epochs.filter(band[0], band[1], method="fir", phase="zero", verbose="ERROR")
    epochs_csd = compute_current_source_density(epochs.copy())
    data = epochs_csd.get_data()[0]  # (n_ch, n_times)
    if smooth_sec and smooth_sec > 0:
        fs = 1.0 / np.median(np.diff(epochs_csd.times))
        k = max(1, int(round(smooth_sec * fs)))
        data = np.vstack([moving_average(ch, k) for ch in data])
    analytic = hilbert(data, axis=-1)
    power = np.abs(analytic) ** 2
    power_mean = power.mean(axis=1)
    return epochs_csd.ch_names, power_mean, epochs_csd.info


def db_ratio(post: np.ndarray, base: np.ndarray) -> np.ndarray:
    """10*log10(post/base) — relative band power change [dB] (unitless)."""
    eps = 1e-12
    base = np.where(base <= 0, eps, base)
    post = np.where(post <= 0, eps, post)
    return 10.0 * np.log10(post / base)


def compute_common_baseline_maps(pre_active_path: Path,
                                 post_sham_path: Path,
                                 post_active_path: Path,
                                 montage: str | None,
                                 sfreq: float | None,
                                 band=(30.0, 45.0)):
    """Return (ch_names, dB_sham_vs_preActive, dB_active_vs_preActive, info) on a COMMON channel set."""
    # Load baseline (pre-active) and both targets
    base_raw = load_raw_edf(pre_active_path, montage, sfreq)
    s_raw    = load_raw_edf(post_sham_path,   montage, sfreq)
    a_raw    = load_raw_edf(post_active_path, montage, sfreq)

    ch_b, p_b, info_b = csd_band_power_per_channel(base_raw, band)
    ch_s, p_s, info_s = csd_band_power_per_channel(s_raw,    band)
    ch_a, p_a, info_a = csd_band_power_per_channel(a_raw,    band)

    # Common channel set across ALL THREE (preserve your canonical order)
    common = [ch for ch in CHANNELS_1005 if ch in ch_b and ch in ch_s and ch in ch_a]
    if len(common) < 4:
        raise RuntimeError(f"Not enough common channels. Common: {common}")

    idx_b = [ch_b.index(ch) for ch in common]
    idx_s = [ch_s.index(ch) for ch in common]
    idx_a = [ch_a.index(ch) for ch in common]

    base_vec = p_b[idx_b]
    sham_vec = p_s[idx_s]
    actv_vec = p_a[idx_a]

    sham_db = db_ratio(sham_vec, base_vec)
    actv_db = db_ratio(actv_vec, base_vec)

    info_sub = mne.pick_info(info_b, [info_b.ch_names.index(ch) for ch in common])
    return common, sham_db, actv_db, info_sub


def _make_plot_info(ch_names, montage_name):
    info = mne.create_info(ch_names=ch_names, sfreq=1000.0, ch_types="eeg")
    info.set_montage(montage_name, match_case=False, on_missing="ignore")
    return info


def plot_two_topomaps(ch_names, sham_db, actv_db,
                      band=(30.0, 45.0),
                      band_symbol="γ",
                      montage_name="standard_1005",
                      save_path: Path | None = None):
    """Left: Sham vs pre-active ("Before TUS"); Right: Active vs pre-active ("After TUS")."""
    info = _make_plot_info(ch_names, montage_name)

    # Shared symmetric limits for fair comparison
    vmin = float(min(sham_db.min(), actv_db.min()))
    vmax = float(max(sham_db.max(), actv_db.max()))
    lim  = max(abs(vmin), abs(vmax)) or 1.0
    vlim = (-lim, lim)

    from matplotlib import gridspec
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[1, 1, 0.05], wspace=0.35)
    ax_left  = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    cax      = fig.add_subplot(gs[0, 2])

    im_left, _ = mne.viz.plot_topomap(sham_db, info, axes=ax_left, vlim=vlim, cmap="RdBu_r",
                                      outlines="head", contours=0, show=False)
    im_right, _ = mne.viz.plot_topomap(actv_db, info, axes=ax_right, vlim=vlim, cmap="RdBu_r",
                                       outlines="head", contours=0, show=False)

    # Titles with Δ and band symbol (δ, θ, α, β, γ)
    ax_left.set_title(f"Before Stimulation")
    ax_right.set_title(f"After Stimulation")

    # Shared colorbar (unitless dB)
    cb = fig.colorbar(im_left, cax=cax)
    cb.set_label(f"Δ {band_symbol} Power [dB]")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# --------------------------- MAIN ----------------------------

def main():
    if save_figs:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Loop over canonical bands and generate the 2-panel figure for each
    for band_symbol, band in BANDS.items():
        ch, sham_db, actv_db, _info = compute_common_baseline_maps(
            pre_active_path, post_sham_path, post_active_path,
            montage=montage_name, sfreq=resample_sfreq, band=band
        )

        save_path = None
        if save_figs:
            lo, hi = int(band[0]), int(band[1])
            save_path = save_dir / f"topomap_{band_symbol}_{lo}-{hi}Hz.png"

        plot_two_topomaps(
            ch, sham_db, actv_db,
            band=band, band_symbol=band_symbol,
            montage_name=montage_name,
            save_path=save_path
        )


if __name__ == "__main__":
    main()
