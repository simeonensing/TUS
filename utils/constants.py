from types import MappingProxyType
from typing import Final

_CANONICAL_BANDS = {
    "delta": (1, 3),
    "theta": (4, 7),
    "alpha": (8, 12),
    "beta":  (13, 30),
    "gamma": (31, 45),
}
CANONICAL_BANDS: Final = MappingProxyType(_CANONICAL_BANDS)  # read-only view

BASELINE_DURATION = 30 # [s]
