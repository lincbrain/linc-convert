"""Entry-points for Dark Field microscopy converter."""
# stdlib
import os.path as op
import warnings
from enum import EnumType, StrEnum
from enum import _EnumDict as EnumDict
from typing import Annotated

# externals
from cyclopts import Parameter

# internals
from linc_convert.cli import main, plain_group
from linc_convert.plain.register import (
    format_to_extension,
    known_converters,
    known_extensions,
)
from linc_convert.utils.opener import stringify_path

known_inp_formats = []
known_out_formats = []
for src, dst in known_converters:
    known_inp_formats.append(src)
    known_out_formats.append(dst)
known_inp_formats = list(set(known_inp_formats))
known_out_formats = list(set(known_out_formats))

_inp_dict = EnumDict()
_inp_dict.update({x for x in known_inp_formats})
inp_hints = EnumType("inp_hints", (StrEnum,), _inp_dict)

_out_dict = EnumDict()
_out_dict.update({x for x in known_inp_formats})
out_hints = EnumType("out_hints", (StrEnum,), _out_dict)


@main.command(name="any", group=plain_group)
def convert(
    inp: str,
    out: str | None = None,
    *,
    inp_hint: inp_hints | None = None,
    out_hint: out_hints | None = None,
    **kwargs: Annotated[dict, Parameter(show=False)],
) -> None:
    """
    Convert between formats while preserving (meta)data as much as possible.

    Parameters
    ----------
    inp
        Path to input file.
    out
        Path to output file.
        If output path not provided, a hint MUST be provided.
    inp_hint
        Input format. Default: guess from extension.
    out_hint
        Output format. Default: guess from extension.
    """
    inp = stringify_path(inp)
    out = stringify_path(out)
    inp_hint = inp_hint or []
    out_hint = out_hint or []
    if not isinstance(inp_hint, list):
        inp_hint = [inp_hint]
    if not isinstance(out_hint, list):
        out_hint = [out_hint]

    # Find type hints from extensions
    for ext, format in known_extensions.items():
        for compression in ('', '.gz', '.bz2'):
            if inp.endswith(ext + compression):
                inp_hint += [format]
            if out and out.endswith(ext + compression):
                out_hint += [format]

    if not inp_hint:
        raise ValueError("Could not guess inoput format from extension")
    if not out_hint:
        raise ValueError("Could not guess output format from extension")

    # Default output name
    if not out:
        out = make_output_path(inp, out_hint[0])

    # Try converter(s)
    for inp_format in inp_hint:
        for out_format in out_hint:
            if (inp_format, out_format) not in known_converters:
                continue
            converter = known_converters[(inp_format, out_format)]
            try:
                return converter(inp, out, **kwargs)
            except Exception:
                warnings.warn(
                    f"Failed to convert from {inp_format} to {out_format}."
                )

    raise RuntimeError("All converters failed.")


def make_output_path(inp: str, format: str) -> str:
    """Create an output path if not provided."""
    base, ext = op.splitext(inp)
    if ext in ('.gz', '.bz2'):
        base, ext = op.splitext(base)
    if ext in ('.zarr', '.tiff') and base.endswith('.ome', '.nii'):
        base, ext = op.splitext(base)

    if "://" in base:
        if base.startswith("file://"):
            base = base[7:]
        else:
            # Input file is remote, make output file local.
            base = op.basename(base)

    return base + format_to_extension[format][0]
