# -----------------------------
# Platform setup (X11 for portability)
# -----------------------------
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"             # force X11
os.environ["XDG_SESSION_TYPE"] = "x11"            # avoid Wayland nag message
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

# -----------------------------
# Imports
# -----------------------------
import sys
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import mne

from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QSizePolicy
from mpl_toolkits.axes_grid1 import make_axes_locatable  # for external RHS colorbar

from utils.helpers import load_pre_post_stim, configure, cmor_tfr
from utils.constants import CANONICAL_BANDS  # {"delta":(1,3), ...}

# -----------------------------
# TFR / preprocessing helpers
# -----------------------------
def _extract_tfr(raw_csd):
    out = cmor_tfr(raw_csd)  # expected (P, freqs) or (P, freqs, times)
    if isinstance(out, tuple) and len(out) == 3:
        P, freqs, times = out
    else:
        P, freqs = out
        times = raw_csd.times
    return np.asarray(P), np.asarray(freqs), np.asarray(times)

def trim_to_common_length(P_pre, t_pre, P_post, t_post, sfreq, expected_minutes=20,
                          prefer_expected=True, warn_label=""):
    n_pre = P_pre.shape[-1]
    n_post = P_post.shape[-1]
    max_diff = int(round(5 * sfreq))
    if abs(n_pre - n_post) > max_diff:
        print(f"[warn] Large length mismatch{f' ({warn_label})' if warn_label else ''}: "
              f"pre={n_pre}, post={n_post} (diff {abs(n_pre-n_post)} samples @ {sfreq:.1f} Hz)")
    if prefer_expected:
        n_expected = int(round(expected_minutes * 60 * sfreq))
        n_target = min(n_expected, n_pre, n_post)
    else:
        n_target = min(n_pre, n_post)
    P_pre_t  = P_pre[..., :n_target]
    P_post_t = P_post[..., :n_target]
    t_ref = t_pre if t_pre.size >= n_target else t_post
    t_t = t_ref[:n_target]
    return P_pre_t, P_post_t, t_t

def compute_shared_clim(A, B, p_lo=5, p_hi=95):
    combined = np.concatenate([A.ravel(), B.ravel()])
    vmin, vmax = np.nanpercentile(combined, [p_lo, p_hi])
    if np.isclose(vmin, vmax):
        pad = 0.1 if vmax == 0 else 0.1 * abs(vmax)
        vmin, vmax = vmin - pad, vmax + pad
    return float(vmin), float(vmax)

# ---- Relative power (percent of total)
def relative_power_per_channel(Ppre_s, Ppre_a, eps=1e-20):
    denom_s = Ppre_s.sum(axis=1, keepdims=True) + eps
    denom_a = Ppre_a.sum(axis=1, keepdims=True) + eps
    RP_sham_pct = (Ppre_s / denom_s) * 100.0
    RP_active_pct = (Ppre_a / denom_a) * 100.0
    return RP_sham_pct, RP_active_pct

def shared_clim_relative_percent(*arrays, lo=1, hi=99):
    flat = np.concatenate([np.asarray(A).ravel() for A in arrays])
    vmin, vmax = np.nanpercentile(flat, [lo, hi])
    vmin = max(0.0, float(vmin))
    vmax = min(100.0, float(vmax if vmax > vmin else vmin + 1.0))
    return vmin, vmax

# -----------------------------
# Block + band aggregation for effect bars
# -----------------------------
def _block_slices(times, block_sec, keep_tail_frac=0.5):
    """Yield time-index slices of length ~block_sec (no overlap)."""
    if times.size < 2:
        yield slice(0, times.size)
        return
    dt = float(np.median(np.diff(times)))
    L  = max(1, int(round(block_sec / dt)))
    start = 0
    N = times.size
    while start + L <= N:
        yield slice(start, start + L)
        start += L
    if start < N and (N - start) >= keep_tail_frac * L:
        yield slice(start, N)

def band_block_means(P, freqs, times, bands_dict, block_sec):
    """
    P: (n_ch, n_freq, n_time)
    Returns: dict[band] -> (n_ch, n_blocks) mean power per block (linear).
    """
    masks = {bn: (freqs >= lo) & (freqs <= hi) for bn, (lo, hi) in bands_dict.items()}
    slices = list(_block_slices(times, block_sec))
    out = {bn: np.zeros((P.shape[0], len(slices)), dtype=float) for bn in bands_dict.keys()}
    for b, slc in enumerate(slices):
        # mean across selected freqs and time samples of the block
        for bn, m in masks.items():
            out[bn][:, b] = P[:, m, :][:, :, slc].mean(axis=(1, 2))
    return out  # each (n_ch, n_blocks)

def effect_with_ci_from_block_ratios(r_sham, r_active, n_boot=2000, ci=95, seed=0):
    """
    r_sham, r_active: 1D arrays of blockwise (post/pre) ratios (linear).
    Returns: (effect_pct, lo_pct, hi_pct), where effect is (ACTIVE vs SHAM) in %.
    """
    eps = 1e-20
    lsh = np.log(np.clip(r_sham,   eps, None))
    lac = np.log(np.clip(r_active, eps, None))

    mdiff = np.median(lac) - np.median(lsh)         # log ratio difference
    effect_pct = (np.exp(mdiff) - 1.0) * 100.0

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        lsh_b = rng.choice(lsh, size=lsh.size, replace=True) if lsh.size else lsh
        lac_b = rng.choice(lac, size=lac.size, replace=True) if lac.size else lac
        md_b  = (np.median(lac_b) - np.median(lsh_b)) if (lac_b.size and lsh_b.size) else 0.0
        boots.append((np.exp(md_b) - 1.0) * 100.0)
    alpha = (100 - ci) / 2
    lo, hi = np.percentile(boots, [alpha, 100 - alpha])
    return float(effect_pct), float(lo), float(hi)

# -----------------------------
# Qt widgets
# -----------------------------
class ChannelBlockWidget(QtWidgets.QWidget):
    """
    One channel = 3-row block:
      Row 1: PRE relative power (%) -> SHAM | ACTIVE      [shared % colorbar]
      Row 2: ΔPower [dB] (Post vs Pre) -> SHAM | ACTIVE    [shared dB colorbar]
      Row 3: Band-wise effect [%] (ACTIVE vs SHAM) with 95% CI error bars (single wide axis)
    The bar row is set to be 2× taller than each scalogram row (configurable).
    """
    def __init__(self,
                 rp_sham_pct, t_sham, rp_active_pct, t_active,      # row 1
                 dsham_db, t_sham2, dactive_db, t_active2,          # row 2
                 bands, eff_vals, ci_lo, ci_hi,                     # row 3 (per-band)
                 freqs, ch_name,
                 vmin_rel, vmax_rel, vmin_db, vmax_db,
                 parent=None, fig_size=(12, 8.2), canvas_min_height_px=700,
                 cmap_rel="viridis", cmap_db="RdBu_r",
                 bar_height_ratio=2.0):  # how tall the bar row is vs one scalogram row
        super().__init__(parent)

        # ---- Figure & layout (GridSpec so last row spans the full width)
        self.fig = plt.figure(figsize=fig_size)
        gs = self.fig.add_gridspec(3, 2, height_ratios=[0.5, 0.5, 1])
        ax00 = self.fig.add_subplot(gs[0, 0])
        ax01 = self.fig.add_subplot(gs[0, 1])
        ax10 = self.fig.add_subplot(gs[1, 0])
        ax11 = self.fig.add_subplot(gs[1, 1])
        ax20 = self.fig.add_subplot(gs[2, :])  # spans both columns
        self.axes = np.array([[ax00, ax01],
                              [ax10, ax11],
                              [ax20, ax20]], dtype=object)  # convenience

        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.setMinimumHeight(int(canvas_min_height_px))
        self.canvas.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(4)
        vbox.addWidget(self.toolbar)
        vbox.addWidget(self.canvas)

        # Extents
        extent_sham1 = [t_sham[0], t_sham[-1], freqs[0], freqs[-1]]
        extent_act1  = [t_active[0], t_active[-1], freqs[0], freqs[-1]]
        extent_sham2 = [t_sham2[0], t_sham2[-1], freqs[0], freqs[-1]]
        extent_act2  = [t_active2[0], t_active2[-1], freqs[0], freqs[-1]]

        # --- Row 1: PRE relative power (% of total)
        im_rp_sham = ax00.imshow(
            rp_sham_pct, aspect="auto", origin="lower",
            extent=extent_sham1, vmin=vmin_rel, vmax=vmax_rel, cmap=cmap_rel
        )
        ax00.set_title("PRE — SHAM (Relative Power)")
        ax00.set_ylabel("Freq (Hz)")

        im_rp_act = ax01.imshow(
            rp_active_pct, aspect="auto", origin="lower",
            extent=extent_act1, vmin=vmin_rel, vmax=vmax_rel, cmap=cmap_rel
        )
        ax01.set_title("PRE — ACTIVE (Relative Power)")

        # Row 1 colorbar (percent) on RHS of ax01
        div_top = make_axes_locatable(ax01)
        cax_top = div_top.append_axes("right", size="3.5%", pad=0.04)
        cbar_top = self.fig.colorbar(im_rp_act, cax=cax_top)
        cbar_top.set_label("Relative Power [% of total]", rotation=90, labelpad=10)

        # ---- Row 2: ΔPower [dB] (Post vs Pre)
        im_dsham = ax10.imshow(
            dsham_db, aspect="auto", origin="lower",
            extent=extent_sham2, vmin=vmin_db, vmax=vmax_db, cmap=cmap_db
        )
        ax10.set_title("SHAM — ΔPower (Post vs Pre)")
        ax10.set_ylabel("Freq (Hz)")
        ax10.set_xlabel("Time (s)")

        im_dact = ax11.imshow(
            dactive_db, aspect="auto", origin="lower",
            extent=extent_act2, vmin=vmin_db, vmax=vmax_db, cmap=cmap_db
        )
        ax11.set_title("ACTIVE — ΔPower (Post vs Pre)")
        ax11.set_xlabel("Time (s)")

        # Row 2 colorbar (dB) on RHS of ax11
        div_bot = make_axes_locatable(ax11)
        cax_bot = div_bot.append_axes("right", size="3.5%", pad=0.04)
        cbar_bot = self.fig.colorbar(im_dact, cax=cax_bot)
        cbar_bot.set_label("Δ Power [dB]", rotation=90, labelpad=10)

        # ---- Row 3: band-wise effect with 95% CI
        x = np.arange(len(bands))
        vals = np.asarray(eff_vals, float)
        lo   = np.asarray(ci_lo,   float)
        hi   = np.asarray(ci_hi,   float)
        err_lo = vals - lo
        err_hi = hi - vals

        bars = ax20.bar(x, vals, width=0.65)
        # error bars in different color + caps; grid on Y
        ax20.errorbar(x, vals, yerr=[err_lo, err_hi], fmt='none',
                      ecolor='tab:gray', elinewidth=1.2, capsize=4, capthick=1.2)
        ax20.grid(True, axis='y', linestyle='--', alpha=0.45)

        ax20.axhline(0, color='k', lw=0.8, alpha=0.5)
        ax20.set_ylabel("Effect [%] (ACTIVE vs SHAM)")
        ax20.set_xticks(x)
        ax20.set_xticklabels(bands)
        ax20.set_title("Band-wise Effect (median block ratio ± 95% CI)")

        # Channel header centered above all panels
        self.fig.suptitle(f"{ch_name}", x=0.5, y=0.99, fontsize=12, fontweight="bold")

        # Margins (room for headers, labels, and RHS colorbars)
        self.fig.subplots_adjust(
            left=0.11, right=0.86, bottom=0.08, top=0.94,
            hspace=0.35, wspace=0.28
        )
        self.canvas.draw_idle()

class ScalogramViewer(QtWidgets.QMainWindow):
    """
    Scrollable list of ChannelBlockWidget, one per channel.
    """
    def __init__(self,
                 RP_sham_pct, t_sham, RP_active_pct, t_active,         # row 1
                 dSHAM_tf, t_sham2, dACTIVE_tf, t_active2,             # row 2
                 band_names, effect_pct, ci_lo_pct, ci_hi_pct,         # row 3
                 freqs, ch_names,
                 parent=None, display_max_time_points=4000,
                 row_height_fraction=0.25, bar_height_ratio=2.0):
        super().__init__(parent)
        self.setWindowTitle("TUS: PRE Relative Power + ΔdB + Band Effects")
        self.setWindowFlag(QtCore.Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, True)

        # Shared scales
        vmin_db,  vmax_db  = compute_shared_clim(dSHAM_tf, dACTIVE_tf)
        vmin_rel, vmax_rel = shared_clim_relative_percent(RP_sham_pct, RP_active_pct)

        # Decimate helper
        def decimate_time(X, target_points, t):
            if X.shape[-1] <= target_points:
                return X, t
            step = max(1, int(np.ceil(X.shape[-1] / target_points)))
            return X[..., ::step], t[::step]

        # Decimate each set separately (they may have different lengths)
        RP_s_d,  t_sham_d = decimate_time(RP_sham_pct,  display_max_time_points, t_sham)
        RP_a_d,  t_act_d  = decimate_time(RP_active_pct, display_max_time_points, t_active)
        dSHAM_d, t_sham2_d = decimate_time(dSHAM_tf,    display_max_time_points, t_sham2)
        dACT_d,  t_act2_d  = decimate_time(dACTIVE_tf,  display_max_time_points, t_active2)

        # Two scalogram rows (each = 1 unit) + bar row (= bar_height_ratio units)
        screen = QtWidgets.QApplication.primaryScreen()
        screen_h_px = screen.geometry().height()
        per_unit_px = max(200, int(screen_h_px * row_height_fraction))  # base height per unit
        block_units = 2.0 + float(bar_height_ratio)
        block_h_px = int(per_unit_px * block_units)

        # Scroll area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        scroll.setWidget(container)

        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Add a ChannelBlockWidget per channel
        for i, ch in enumerate(ch_names):
            w = ChannelBlockWidget(
                rp_sham_pct=RP_s_d[i], t_sham=t_sham_d,
                rp_active_pct=RP_a_d[i], t_active=t_act_d,
                dsham_db=dSHAM_d[i], t_sham2=t_sham2_d,
                dactive_db=dACT_d[i], t_active2=t_act2_d,
                bands=band_names,
                eff_vals=effect_pct[i], ci_lo=ci_lo_pct[i], ci_hi=ci_hi_pct[i],
                freqs=freqs, ch_name=ch,
                vmin_rel=vmin_rel, vmax_rel=vmax_rel,
                vmin_db=vmin_db,   vmax_db=vmax_db,
                fig_size=(12, 8.2),
                canvas_min_height_px=block_h_px,
                bar_height_ratio=bar_height_ratio
            )
            layout.addWidget(w)

            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.HLine)
            line.setFrameShadow(QtWidgets.QFrame.Sunken)
            layout.addWidget(line)

        layout.addItem(QtWidgets.QSpacerItem(
            20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        ))
        self.setCentralWidget(scroll)
        self.resize(1200, 980)

# -----------------------------
# Main (compute + Qt viewer)
# -----------------------------
def main():
    # ---- Configure and session paths
    (pre_sham_path, post_sham_path, pre_active_path, post_active_path,
     montage, window_dur, overlap, channels) = configure("config.yaml")

    sessions = {
        "pre_sham": pre_sham_path,
        "post_sham": post_sham_path,
        "pre_active": pre_active_path,
        "post_active": post_active_path,
    }

    # ---- Compute CSD -> TFR for each session
    power_full = {}
    freqs_by_key = {}
    times_by_key = {}
    ch_names = None
    sfreq = None

    for key, session_path in sessions.items():
        raw = load_pre_post_stim(session_path, channels, montage)
        raw_csd = mne.preprocessing.compute_current_source_density(
            raw, lambda2=1e-5, stiffness=4.0, n_legendre_terms=50
        )
        P, freqs, times = _extract_tfr(raw_csd)
        power_full[key] = P
        freqs_by_key[key] = freqs
        times_by_key[key] = times
        if ch_names is None:
            ch_names = list(raw_csd.ch_names)
        if sfreq is None:
            sfreq = float(raw_csd.info["sfreq"])

    assert sfreq is not None and sfreq > 0, "Could not determine sampling rate (sfreq)."
    # freq grids match within condition
    assert np.allclose(freqs_by_key["pre_sham"],   freqs_by_key["post_sham"]),   "Freq grid mismatch in SHAM"
    assert np.allclose(freqs_by_key["pre_active"], freqs_by_key["post_active"]), "Freq grid mismatch in ACTIVE"
    freqs = freqs_by_key["pre_sham"]

    # ---- Trim to identical lengths (within each condition)
    Ppre_s, Ppost_s, t_sham = trim_to_common_length(
        power_full["pre_sham"], times_by_key["pre_sham"],
        power_full["post_sham"], times_by_key["post_sham"],
        sfreq=sfreq, expected_minutes=20, prefer_expected=True, warn_label="SHAM"
    )
    Ppre_a, Ppost_a, t_active = trim_to_common_length(
        power_full["pre_active"], times_by_key["pre_active"],
        power_full["post_active"], times_by_key["post_active"],
        sfreq=sfreq, expected_minutes=20, prefer_expected=True, warn_label="ACTIVE"
    )

    # ---- PRE relative power per channel (row 1)
    eps = 1e-20
    RP_sham_ch_pct, RP_active_ch_pct = relative_power_per_channel(Ppre_s, Ppre_a, eps=eps)

    # ---- ΔdB (post/pre) per condition (row 2)
    dSHAM_tf   = 10.0 * np.log10((Ppost_s + eps) / (Ppre_s + eps))
    dACTIVE_tf = 10.0 * np.log10((Ppost_a + eps) / (Ppre_a + eps))

    # ---- Band-wise block ratios for error bars (row 3)
    band_names = list(CANONICAL_BANDS.keys())
    # mean power per block per band (linear)
    preS_blocks  = band_block_means(Ppre_s,  freqs, t_sham,   CANONICAL_BANDS, window_dur)
    postS_blocks = band_block_means(Ppost_s, freqs, t_sham,   CANONICAL_BANDS, window_dur)
    preA_blocks  = band_block_means(Ppre_a,  freqs, t_active, CANONICAL_BANDS, window_dur)
    postA_blocks = band_block_means(Ppost_a, freqs, t_active, CANONICAL_BANDS, window_dur)

    # blockwise ratios (post/pre), shape: (n_ch, n_blocks_band)
    ratios_sham   = {bn: (postS_blocks[bn] + eps) / (preS_blocks[bn] + eps) for bn in band_names}
    ratios_active = {bn: (postA_blocks[bn] + eps) / (preA_blocks[bn] + eps) for bn in band_names}

    # effect + 95% CI per channel, per band
    n_ch   = len(ch_names)
    n_band = len(band_names)
    effect_pct = np.zeros((n_ch, n_band), float)
    ci_lo_pct  = np.zeros((n_ch, n_band), float)
    ci_hi_pct  = np.zeros((n_ch, n_band), float)

    for bi, bn in enumerate(band_names):
        for ci in range(n_ch):
            r_sh = ratios_sham[bn][ci, :]
            r_ac = ratios_active[bn][ci, :]
            v, lo, hi = effect_with_ci_from_block_ratios(
                r_sh, r_ac, n_boot=2000, ci=95, seed=ci*97 + bi
            )
            effect_pct[ci, bi] = v
            ci_lo_pct[ci, bi]  = lo
            ci_hi_pct[ci, bi]  = hi

    # ---- Launch the PyQt viewer
    app = QtWidgets.QApplication(sys.argv)
    print("[qt] platform:", app.platformName())  # should print 'xcb'

    viewer = ScalogramViewer(
        RP_sham_pct=RP_sham_ch_pct, t_sham=t_sham,
        RP_active_pct=RP_active_ch_pct, t_active=t_active,
        dSHAM_tf=dSHAM_tf, t_sham2=t_sham,
        dACTIVE_tf=dACTIVE_tf, t_active2=t_active,
        band_names=band_names, effect_pct=effect_pct,
        ci_lo_pct=ci_lo_pct, ci_hi_pct=ci_hi_pct,
        freqs=freqs, ch_names=ch_names,
        display_max_time_points=4000,
        row_height_fraction=0.25,
        bar_height_ratio=2.0
    )
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
