"""Entry-points for Dark Field microscopy converter."""

from cyclopts import App

from linc_convert.cli import main

help = "Converters for Dark Field microscopy"
df = App(name="df", help=help)
main.command(df)
