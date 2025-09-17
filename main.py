#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==========================================
# Platform setup (X11 for portability)
# ==========================================
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
# Let NumPy/BLAS use threads; avoid fighting with Python-level parallelism
os.environ.setdefault("OMP_NUM_THREADS", "0")
os.environ.setdefault("MKL_NUM_THREADS", "0")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "0")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "0")

# ==========================================
# Imports
# ==========================================
import sys
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")

import mne

from PyQt5 import QtWidgets

# Progress bars (white)
from functools import partial
try:
    from tqdm.auto import tqdm
except Exception:  # minimal fallback
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))
TQDM = partial(tqdm, dynamic_ncols=True, leave=False, file=sys.stdout, colour="white")

# Your utilities
from utils.helpers import load_pre_post_stim, configure, _extract_tfr, trim_to_common_length, compute_shared_clim, \
    relative_power_per_channel, shared_clim_relative_percent, _block_slices, band_block_means, \
    effect_with_ci_from_block_ratios, plv_band_block_values, effect_with_ci_from_plv_deltas, plv_maps_from_blocks, \
    shared_clim_plv, spectral_entropy_blocks, se_maps_from_blocks, shared_clim_unit_interval, msc_blocks, \
    msc_maps_from_blocks, _get_channel_xy, percent_change_map, mats_effect_ratio_of_ratios, symmetric_limits_from_mats, \
    SinglePanelViewer
from utils.constants import CANONICAL_BANDS  # {"delta":(1,3),...}

from utils.progress import TQDM

def main():
    # ---- Config
    (pre_sham_path, post_sham_path, pre_active_path, post_active_path,
     montage, window_dur, overlap, channels) = configure("config.yaml")

    sessions = {
        "pre_sham": pre_sham_path,
        "post_sham": post_sham_path,
        "pre_active": pre_active_path,
        "post_active": post_active_path,
    }

    power_full = {}
    freqs_by_key = {}
    times_by_key = {}
    raws_csd = {}
    ch_names = None
    sfreq = None

    # ---- Load + CSD + TFR
    for key, session_path in TQDM(list(sessions.items()), desc="Loading sessions + TFR", unit="sess"):
        raw = load_pre_post_stim(session_path, channels, montage)
        raw_csd = mne.preprocessing.compute_current_source_density(
            raw, lambda2=1e-5, stiffness=4.0, n_legendre_terms=50
        )
        try:  # keep montage
            raw_csd.set_montage(raw.get_montage(), on_missing="ignore", match_case=False)
        except Exception:
            pass

        P, freqs, times = _extract_tfr(raw_csd)
        power_full[key] = P
        freqs_by_key[key] = freqs
        times_by_key[key] = times
        raws_csd[key] = raw_csd

        if ch_names is None:
            ch_names = list(raw_csd.ch_names)
        if sfreq is None:
            sfreq = float(raw_csd.info["sfreq"])

    assert sfreq is not None and sfreq > 0
    assert np.allclose(freqs_by_key["pre_sham"],   freqs_by_key["post_sham"])
    assert np.allclose(freqs_by_key["pre_active"], freqs_by_key["post_active"])
    freqs = freqs_by_key["pre_sham"]

    # ---- Trim within condition
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

    # ==========================================
    # POWER maps + effects (with progress)
    # ==========================================
    eps = 1e-20
    with TQDM(total=8, desc="POWER maps + effects — setup", unit="step") as pbar:
        RP_sham_ch_pct, RP_active_ch_pct = relative_power_per_channel(Ppre_s, Ppre_a, eps=eps); pbar.update(1)
        dSHAM_tf   = 10.0 * np.log10((Ppost_s + eps) / (Ppre_s + eps)); pbar.update(1)
        dACTIVE_tf = 10.0 * np.log10((Ppost_a + eps) / (Ppre_a + eps)); pbar.update(1)

        band_names = list(CANONICAL_BANDS.keys())
        preS_blocks,  _ = band_block_means(Ppre_s, freqs, t_sham, CANONICAL_BANDS, window_dur); pbar.update(1)
        postS_blocks, _ = band_block_means(Ppost_s, freqs, t_sham, CANONICAL_BANDS, window_dur); pbar.update(1)
        preA_blocks,  _ = band_block_means(Ppre_a, freqs, t_active, CANONICAL_BANDS, window_dur); pbar.update(1)
        postA_blocks, _ = band_block_means(Ppost_a, freqs, t_active, CANONICAL_BANDS, window_dur); pbar.update(1)

        ratios_sham   = {bn: (postS_blocks[bn] + eps) / (preS_blocks[bn] + eps) for bn in band_names}
        ratios_active = {bn: (postA_blocks[bn] + eps) / (preA_blocks[bn] + eps) for bn in band_names}; pbar.update(1)

        n_ch   = len(ch_names); n_band = len(band_names)
        power_effect_pct = np.zeros((n_ch, n_band), float)
        power_ci_lo_pct  = np.zeros((n_ch, n_band), float)
        power_ci_hi_pct  = np.zeros((n_ch, n_band), float)

    for bi, bn in enumerate(TQDM(band_names, desc="POWER effects — bands", unit="band")):
        for ci in range(n_ch):
            r_sh = ratios_sham[bn][ci, :]
            r_ac = ratios_active[bn][ci, :]
            v, lo, hi = effect_with_ci_from_block_ratios(r_sh, r_ac, n_boot=2000, ci=95, seed=ci * 97 + bi)
            power_effect_pct[ci, bi] = v
            power_ci_lo_pct[ci, bi]  = lo
            power_ci_hi_pct[ci, bi]  = hi

    # ==========================================
    # PLV maps + effects
    # ==========================================
    raw_pre_s  = raws_csd["pre_sham"].copy().crop(tmin=t_sham[0],  tmax=t_sham[-1],  include_tmax=True)
    raw_post_s = raws_csd["post_sham"].copy().crop(tmin=t_sham[0],  tmax=t_sham[-1],  include_tmax=True)
    raw_pre_a  = raws_csd["pre_active"].copy().crop(tmin=t_active[0], tmax=t_active[-1], include_tmax=True)
    raw_post_a = raws_csd["post_active"].copy().crop(tmin=t_active[0], tmax=t_active[-1], include_tmax=True)

    def block_mid_times(times, block_sec):
        sl = list(_block_slices(times, block_sec))
        if not sl:
            return np.array([times.mean()]), []
        mids = [times[s].mean() for s in sl]
        return np.asarray(mids, float), sl

    tb_sham,  _ = block_mid_times(t_sham,  window_dur)
    tb_active, _ = block_mid_times(t_active,window_dur)
    tb2_sham, _  = block_mid_times(t_sham,  window_dur)
    tb2_active,_ = block_mid_times(t_active,window_dur)

    plv_pre_s_dict,  _ = plv_band_block_values(raw_pre_s, CANONICAL_BANDS, t_sham, window_dur,
                                               show_progress=True, desc="PLV PRE SHAM")
    plv_post_s_dict, _ = plv_band_block_values(raw_post_s, CANONICAL_BANDS, t_sham, window_dur,
                                               show_progress=True, desc="PLV POST SHAM")
    plv_pre_a_dict,  _ = plv_band_block_values(raw_pre_a, CANONICAL_BANDS, t_active, window_dur,
                                               show_progress=True, desc="PLV PRE ACTIVE")
    plv_post_a_dict, _ = plv_band_block_values(raw_post_a, CANONICAL_BANDS, t_active, window_dur,
                                               show_progress=True, desc="PLV POST ACTIVE")

    plv_pre_sham_map,  plv_band_names = plv_maps_from_blocks(plv_pre_s_dict)
    plv_pre_active_map,_             = plv_maps_from_blocks(plv_pre_a_dict)

    def dplv_map(pre_dict, post_dict):
        bnames = list(pre_dict.keys())
        n_ch = pre_dict[bnames[0]].shape[0]; n_b = len(bnames)
        n_blk = min(pre_dict[bnames[0]].shape[1], post_dict[bnames[0]].shape[1])
        out = np.zeros((n_ch, n_b, n_blk), float)
        for bi, bn in enumerate(bnames):
            out[:, bi, :] = post_dict[bn][:, :n_blk] - pre_dict[bn][:, :n_blk]
        return out

    dplv_sham_map   = dplv_map(plv_pre_s_dict, plv_post_s_dict)
    dplv_active_map = dplv_map(plv_pre_a_dict, plv_post_a_dict)

    plv_effect_pp = np.zeros((n_ch, n_band), float)
    plv_ci_lo_pp  = np.zeros((n_ch, n_band), float)
    plv_ci_hi_pp  = np.zeros((n_ch, n_band), float)
    for bi, bn in enumerate(TQDM(plv_band_names, desc="PLV effects — bands", unit="band")):
        for ci in range(n_ch):
            d_sh = (plv_post_s_dict[bn][ci, :] - plv_pre_s_dict[bn][ci, :])
            d_ac = (plv_post_a_dict[bn][ci, :] - plv_pre_a_dict[bn][ci, :])
            v, lo, hi = effect_with_ci_from_plv_deltas(d_sh, d_ac, n_boot=2000, ci=95, seed=ci * 137 + bi)
            plv_effect_pp[ci, bi] = v
            plv_ci_lo_pp[ci, bi]  = lo
            plv_ci_hi_pp[ci, bi]  = hi

    # ==========================================
    # Spectral Entropy maps + effects
    # ==========================================
    se_pre_s_dict,  _ = spectral_entropy_blocks(Ppre_s, freqs, t_sham, CANONICAL_BANDS, window_dur,
                                                show_progress=True, desc="SE PRE SHAM")
    se_post_s_dict, _ = spectral_entropy_blocks(Ppost_s, freqs, t_sham, CANONICAL_BANDS, window_dur,
                                                show_progress=True, desc="SE POST SHAM")
    se_pre_a_dict,  _ = spectral_entropy_blocks(Ppre_a, freqs, t_active, CANONICAL_BANDS, window_dur,
                                                show_progress=True, desc="SE PRE ACTIVE")
    se_post_a_dict, _ = spectral_entropy_blocks(Ppost_a, freqs, t_active, CANONICAL_BANDS, window_dur,
                                                show_progress=True, desc="SE POST ACTIVE")

    se_pre_sham_map,  se_band_names = se_maps_from_blocks(se_pre_s_dict)
    se_pre_active_map,_             = se_maps_from_blocks(se_pre_a_dict)

    def dse_map(pre_dict, post_dict):
        bnames = list(pre_dict.keys())
        n_ch = pre_dict[bnames[0]].shape[0]; n_b = len(bnames)
        n_blk = min(pre_dict[bnames[0]].shape[1], post_dict[bnames[0]].shape[1])
        out = np.zeros((n_ch, n_b, n_blk), float)
        for bi, bn in enumerate(bnames):
            out[:, bi, :] = post_dict[bn][:, :n_blk] - pre_dict[bn][:, :n_blk]
        return out

    dse_sham_map   = dse_map(se_pre_s_dict, se_post_s_dict)
    dse_active_map = dse_map(se_pre_a_dict, se_post_a_dict)

    se_effect = np.zeros((n_ch, n_band), float)
    se_ci_lo  = np.zeros((n_ch, n_band), float)
    se_ci_hi  = np.zeros((n_ch, n_band), float)
    for bi, bn in enumerate(TQDM(se_band_names, desc="SE effects — bands", unit="band")):
        for ci in range(n_ch):
            r_sh = (se_post_s_dict[bn][ci, :] + eps) / (se_pre_s_dict[bn][ci, :] + eps)
            r_ac = (se_post_a_dict[bn][ci, :] + eps) / (se_pre_a_dict[bn][ci, :] + eps)
            v, lo, hi = effect_with_ci_from_block_ratios(r_sh, r_ac, n_boot=2000, ci=95, seed=ci * 211 + bi)
            se_effect[ci, bi] = v
            se_ci_lo[ci, bi]  = lo
            se_ci_hi[ci, bi]  = hi

    # ==========================================
    # MSC maps + effects + ego EFFECT matrices + blockwise mats (for bars)
    # ==========================================
    (msc_pre_s_blocks,  msc_pre_s_mats,  tb_msc_sham,   _,
     msc_pre_s_blocks_mats) = msc_blocks(
        raws_csd["pre_sham"],  t_sham,   CANONICAL_BANDS, window_dur,
        show_progress=True, desc="MSC PRE SHAM", return_block_mats=True)

    (msc_post_s_blocks, msc_post_s_mats, tb2_msc_sham,  _,
     msc_post_s_blocks_mats) = msc_blocks(
        raws_csd["post_sham"], t_sham,   CANONICAL_BANDS, window_dur,
        show_progress=True, desc="MSC POST SHAM", return_block_mats=True)

    (msc_pre_a_blocks,  msc_pre_a_mats,  tb_msc_active, _,
     msc_pre_a_blocks_mats) = msc_blocks(
        raws_csd["pre_active"], t_active, CANONICAL_BANDS, window_dur,
        show_progress=True, desc="MSC PRE ACTIVE", return_block_mats=True)

    (msc_post_a_blocks, msc_post_a_mats, tb2_msc_active,_,
     msc_post_a_blocks_mats) = msc_blocks(
        raws_csd["post_active"], t_active, CANONICAL_BANDS, window_dur,
        show_progress=True, desc="MSC POST ACTIVE", return_block_mats=True)

    msc_pre_sham_map,  msc_band_names = msc_maps_from_blocks(msc_pre_s_blocks)
    msc_pre_active_map,_             = msc_maps_from_blocks(msc_pre_a_blocks)
    dmsc_sham_map   = percent_change_map(msc_pre_s_blocks, msc_post_s_blocks)
    dmsc_active_map = percent_change_map(msc_pre_a_blocks, msc_post_a_blocks)

    msc_effect = np.zeros((n_ch, n_band), float)
    msc_ci_lo  = np.zeros((n_ch, n_band), float)
    msc_ci_hi  = np.zeros((n_ch, n_band), float)
    for bi, bn in enumerate(TQDM(msc_band_names, desc="MSC effects — bands", unit="band")):
        for ci in range(n_ch):
            r_sh = (msc_post_s_blocks[bn][ci, :] + eps) / (msc_pre_s_blocks[bn][ci, :] + eps)
            r_ac = (msc_post_a_blocks[bn][ci, :] + eps) / (msc_pre_a_blocks[bn][ci, :] + eps)
            v, lo, hi = effect_with_ci_from_block_ratios(r_sh, r_ac, n_boot=2000, ci=95, seed=ci * 307 + bi)
            msc_effect[ci, bi] = v
            msc_ci_lo[ci, bi]  = lo
            msc_ci_hi[ci, bi]  = hi

    # ---- Ego-net EFFECT adjacency (Active vs Sham), per band
    ego_effect_mats, ego_band_names = mats_effect_ratio_of_ratios(
        msc_pre_s_mats, msc_post_s_mats, msc_pre_a_mats, msc_post_a_mats
    )
    edge_vmin, edge_vmax = symmetric_limits_from_mats(ego_effect_mats)

    # ---- Ego-net positions (EEG1005/recording montage)
    ego_pos  = _get_channel_xy(raws_csd["pre_active"].info, ch_names, prefer_montage="standard_1005")
    print(f"[ego-net] positions: valid={np.isfinite(ego_pos).all(axis=1).sum()}/{len(ch_names)} "
          f"unique={np.unique(np.round(ego_pos, 3), axis=0).shape[0]}")

    # ---- Color limits
    vmin_db,  vmax_db  = compute_shared_clim(dSHAM_tf, dACTIVE_tf)
    vmin_rel, vmax_rel = shared_clim_relative_percent(RP_sham_ch_pct, RP_active_ch_pct)
    vmin_plv, vmax_plv = shared_clim_plv(plv_pre_sham_map, plv_pre_active_map)
    vmin_dplv, vmax_dplv = compute_shared_clim(dplv_sham_map, dplv_active_map, p_lo=5, p_hi=95)
    vmin_se,  vmax_se  = shared_clim_unit_interval(se_pre_sham_map, se_pre_active_map)
    vmin_dse, vmax_dse = compute_shared_clim(dse_sham_map, dse_active_map, p_lo=5, p_hi=95)
    vmin_msc, vmax_msc = shared_clim_unit_interval(msc_pre_sham_map, msc_pre_active_map)
    vmin_dmsc,vmax_dmsc= compute_shared_clim(dmsc_sham_map, dmsc_active_map, p_lo=5, p_hi=95)

    # ---- Decimate power maps for UI speed
    def decimate_time(X, target_points, t):
        if X.shape[-1] <= target_points:
            return X, t
        step = max(1, int(np.ceil(X.shape[-1] / target_points)))
        return X[..., ::step], t[::step]

    display_max_time_points = 4000
    RP_s_d,  t_sham_d  = decimate_time(RP_sham_ch_pct,  display_max_time_points, t_sham)
    RP_a_d,  t_act_d   = decimate_time(RP_active_ch_pct, display_max_time_points, t_active)
    dSHAM_d, t_sham2_d = decimate_time(dSHAM_tf,         display_max_time_points, t_sham)
    dACT_d,  t_act2_d  = decimate_time(dACTIVE_tf,       display_max_time_points, t_active)

    # ---- Bundle data for the viewer
    data = {
        "power": {
            "rp_s": RP_s_d, "rp_a": RP_a_d, "t_sham": t_sham_d, "t_active": t_act_d,
            "dS": dSHAM_d, "t_sham2": t_sham2_d, "dA": dACT_d, "t_active2": t_act2_d,
            "bands": band_names, "eff": power_effect_pct, "lo": power_ci_lo_pct, "hi": power_ci_hi_pct,
        },
        "plv": {
            "pre_s": plv_pre_sham_map, "pre_a": plv_pre_active_map,
            "d_s": dplv_sham_map, "d_a": dplv_active_map,
            "tb_s": tb_sham, "tb_a": tb_active, "tb2_s": tb2_sham, "tb2_a": tb2_active,
            "bands": plv_band_names, "eff": plv_effect_pp, "lo": plv_ci_lo_pp, "hi": plv_ci_hi_pp,
        },
        "se": {
            "pre_s": se_pre_sham_map, "pre_a": se_pre_active_map,
            "d_s": dse_sham_map, "d_a": dse_active_map,
            "tb_s": tb_sham, "tb_a": tb_active, "tb2_s": tb2_sham, "tb2_a": tb2_active,
            "bands": se_band_names, "eff": se_effect, "lo": se_ci_lo, "hi": se_ci_hi,
        },
        "msc": {
            "pre_s": msc_pre_sham_map, "pre_a": msc_pre_active_map,
            "d_s": dmsc_sham_map, "d_a": dmsc_active_map,
            "tb_s": tb_msc_sham, "tb_a": tb_msc_active, "tb2_s": tb2_msc_sham, "tb2_a": tb2_msc_active,
            "bands": msc_band_names, "eff": msc_effect, "lo": msc_ci_lo, "hi": msc_ci_hi,
        }
    }

    limits = dict(
        vmin_rel=vmin_rel, vmax_rel=vmax_rel, vmin_db=vmin_db, vmax_db=vmax_db,
        vmin_plv=vmin_plv, vmax_plv=vmax_plv, vmin_dplv=vmin_dplv, vmax_dplv=vmax_dplv,
        vmin_se=vmin_se, vmax_se=vmax_se, vmin_dse=vmin_dse, vmax_dse=vmax_dse,
        vmin_msc=vmin_msc, vmax_msc=vmax_msc, vmin_dmsc=vmin_dmsc, vmax_dmsc=vmax_dmsc
    )

    ego = dict(
        pos=_get_channel_xy(raws_csd["pre_active"].info, ch_names, prefer_montage="standard_1005"),
        bands=ego_band_names,
        mats_effect=ego_effect_mats,   # Active vs Sham effect Δ%
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        block_mats=dict(  # for Ego bars
            pre_s=msc_pre_s_blocks_mats,
            post_s=msc_post_s_blocks_mats,
            pre_a=msc_pre_a_blocks_mats,
            post_a=msc_post_a_blocks_mats,
        ),
    )

    # ---- Launch GUI
    app = QtWidgets.QApplication(sys.argv)
    print("[qt] platform:", app.platformName())
    ui_sizes = {"fig": (12.6, 8.4)}
    viewer = SinglePanelViewer(ch_names, freqs, data, limits, ego, ui_sizes)
    viewer.show()
    # Optionally start maximized:
    # viewer.setWindowState(viewer.windowState() | QtCore.Qt.WindowMaximized)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
