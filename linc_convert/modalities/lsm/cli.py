"""Entry-points for Dark Field microscopy converter."""

from cyclopts import App

from linc_convert.cli import main, modalities_group

help = "Converters for Light Sheet Microscopy"
lsm = App(name="lsm", help=help, group=modalities_group)
main.command(lsm)
