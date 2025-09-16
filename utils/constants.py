from types import MappingProxyType
from typing import Final

_CANONICAL_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (12, 30),
    "gamma": (30, 45),
}
CANONICAL_BANDS: Final = MappingProxyType(_CANONICAL_BANDS)  # read-only view

BASELINE_DURATION = 30 # [s]
