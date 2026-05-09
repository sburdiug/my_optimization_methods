import sys
from pathlib import Path

import numpy as np

WORK_DIR = Path(__file__).resolve().parents[1]
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

from golden_section_method import golden_section_method
from powell_method import powell_method
from sven import sven_method


def sven_delta(x, s, alpha: float) -> float:
    if alpha <= 0:
        raise ValueError("alpha має бути додатним.")

    x_norm = float(np.linalg.norm(np.asarray(x, dtype=float)))
    s_norm = float(np.linalg.norm(np.asarray(s, dtype=float)))
    if s_norm <= 1e-14:
        return 0.0

    if x_norm <= 1e-14:
        x_norm = 1.0

    return float(alpha * x_norm / s_norm)


def sven_interval(phi, delta: float = 0.01, x0: float = 0.0, max_iter: int = 60):
    if delta <= 0:
        raise ValueError("delta має бути додатним.")
    if max_iter <= 0:
        raise ValueError("max_iter має бути додатним.")

    interval, _history = sven_method(phi, float(x0), float(delta), max_iter=int(max_iter), verbose=False)
    a, b = interval
    a = float(np.maximum(0.0, np.minimum(a, b)))
    b = float(np.maximum(0.0, np.maximum(interval[0], interval[1])))

    if np.isclose(a, b):
        b = a + float(delta)

    return a, b


def golden_section_search(phi, a: float, b: float, eps: float = 1e-3, max_iter: int = 300):
    if eps <= 0:
        raise ValueError("eps має бути додатним.")
    if max_iter <= 0:
        raise ValueError("max_iter має бути додатним.")

    interval, _history = golden_section_method(
        phi,
        float(a),
        float(b),
        eps=float(eps),
        max_iter=int(max_iter),
        verbose=False,
    )
    left, right = interval
    return float(0.5 * (left + right))


def dsk_powell_search(phi, a: float, b: float, eps: float = 1e-3, max_iter: int = 300):
    if eps <= 0:
        raise ValueError("eps має бути додатним.")
    if max_iter <= 0:
        raise ValueError("max_iter має бути додатним.")

    lambda_opt, _info = powell_method(
        phi,
        float(a),
        float(b),
        eps=float(eps),
        max_iter=int(max_iter),
        verbose=False,
    )
    return float(lambda_opt)


def line_search(phi, method: str, a: float, b: float, eps: float = 1e-3):
    if method == "golden":
        return golden_section_search(phi, a, b, eps=eps)
    if method == "dsk_powell":
        return dsk_powell_search(phi, a, b, eps=eps)
    raise ValueError("line_search_method має бути 'golden' або 'dsk_powell'.")
