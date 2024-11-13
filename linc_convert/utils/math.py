"""Math utilities."""

import math
from numbers import Number


def ceildiv(x: Number, y: Number) -> int:
    """Ceil of ratio of two numbers."""
    return int(math.ceil(x / y))


def floordiv(x: Number, y: Number) -> int:
    """Floor of ratio of two numbers."""
    return int(math.floor(x / y))
