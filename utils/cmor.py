import numpy as np
import mne
from mne.time_frequency import tfr_array_morlet

def decompose_epochs_vectorized(
    epochs: mne.Epochs,
    bands: dict[str, tuple[float, float]],
    *,
    freqs: np.ndarray | None = None,        # frequency grid (Hz)
    n_cycles: float | np.ndarray | None = None,
    decim: int = 1,
    output: str = "power",                  # 'power' | 'complex' | 'phase'
    dtype: np.dtype = np.float32,           # reduce memory
    average: str | None = None,             # None | 'time' | 'epochs' | 'time_epochs'
):
    """
    Vectorized band decomposition using complex Morlet wavelets.

    Returns
    -------
    band_power : np.ndarray
        Shape (n_epochs, n_channels, n_bands, n_times) if average is None.
        If average == 'time'       -> (n_epochs, n_channels, n_bands)
        If average == 'epochs'     -> (n_channels, n_bands, n_times)
        If average == 'time_epochs'-> (n_channels, n_bands)
    band_names : list[str]
        Order of bands along the 3rd axis.
    freqs_used : np.ndarray
        Frequencies used (Hz).
    """
    X = epochs.get_data().astype(dtype, copy=False)         # (E, C, T)
    sfreq = float(epochs.info["sfreq"])

    # Frequency grid (linear is fine; use logspace if you prefer)
    if freqs is None:
        freqs = np.arange(1.0, 46.0, 1.0, dtype=float)      # 1..45 Hz
    freqs = np.asarray(freqs, dtype=float)

    # Cycles per freq: variable cycles -> better TF tradeoff
    if n_cycles is None:
        n_cycles = np.linspace(3.0, 10.0, freqs.size, dtype=float)

    # Compute TFR over all epochs/channels at once
    # Output shapes: (E, C, F, T_out)
    tfr = tfr_array_morlet(
        X, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
        output=output, decim=decim, zero_mean=True, n_jobs=1,
    )

    # Convert to something aggregatable
    if output == "power":
        tfr_mag = tfr.astype(dtype, copy=False)
    elif output == "complex":
        # convert to power (magnitude^2); or np.abs(tfr) for amplitude
        tfr_mag = (tfr.real.astype(dtype, copy=False)**2 + tfr.imag.astype(dtype, copy=False)**2)
    elif output == "phase":
        # phase does not have "power"; you probably want amplitude/PLV instead
        tfr_mag = np.angle(tfr).astype(dtype, copy=False)
    else:
        raise ValueError("output must be one of: 'power' | 'complex' | 'phase'")

    # Build band weights matrix W of shape (B, F)
    band_names = list(bands.keys())
    B, F = len(band_names), freqs.size
    W = np.zeros((B, F), dtype=dtype)
    for i, name in enumerate(band_names):
        lo, hi = bands[name]
        m = (freqs >= lo) & (freqs <= hi)
        if not np.any(m):
            continue
        # Uniform average within the band
        W[i, m] = 1.0 / m.sum()

    # Aggregate across frequency axis with one tensor op
    # tfr_mag: (E, C, F, T), W: (B, F)
    # tensordot over F -> (E, C, T, B), then move B to axis 2
    band_agg = np.tensordot(tfr_mag, W, axes=([2], [1]))    # (E, C, T, B)
    band_agg = np.moveaxis(band_agg, -1, 2).astype(dtype, copy=False)  # (E, C, B, T)

    # Optional averaging
    if average is not None:
        if average == "time":
            band_agg = band_agg.mean(axis=-1)               # (E, C, B)
        elif average == "epochs":
            band_agg = band_agg.mean(axis=0)                # (C, B, T)
        elif average == "time_epochs":
            band_agg = band_agg.mean(axis=(0, -1))          # (C, B)
        else:
            raise ValueError("average must be one of None|'time'|'epochs'|'time_epochs'")

    return band_agg, band_names, freqs