"""Webknossos annotation converters."""


try:
    import wkw as _  # noqa: F401

    __all__ = ["cli", "webknossos_annotation"]
    from . import cli, webknossos_annotation
except ImportError:
    pass
