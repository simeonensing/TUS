# utils/progress.py
import sys
from functools import partial
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))
TQDM = partial(tqdm, dynamic_ncols=True, leave=False, file=sys.stdout, colour="white")
