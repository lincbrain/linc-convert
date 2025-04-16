"""Entry-points for Dark Field microscopy converter."""

from cyclopts import App

from linc_convert.cli import main, modalities_group

help = "Converters for Dark Field microscopy"
df = App(name="df", help=help, group=modalities_group)
main.command(df)
