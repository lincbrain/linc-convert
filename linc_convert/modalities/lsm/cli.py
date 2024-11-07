"""Entry-points for Dark Field microscopy converter."""

from cyclopts import App

from linc_convert.cli import main

help = "Converters for Light Sheet Microscopy"
lsm = App(name="lsm", help=help)
main.command(lsm)
