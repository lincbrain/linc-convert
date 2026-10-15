"""Dark Field microscopy converters."""

from importlib.util import find_spec

# glymur is imported lazily: importing it (via lxml) re-enables the GIL on
# free-threaded Python, which would slow down every other converter.
if find_spec("glymur"):
    __all__ = ["cli", "multi_slice", "single_slice"]

    from . import cli, multi_slice, single_slice
