
"""

Revit utilities
"""

_TOLERANCE = 128 # 1/128 inch


def round_inches(x, tol=_TOLERANCE):
    return round(x * tol) / tol
