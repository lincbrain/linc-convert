"""Entry-points for Webknossos annotation converter."""

from cyclopts import App

from linc_convert.cli import main, modalities_group

help = "Converters for Webknossos annotation"
wk = App(name="wk", help=help, group=modalities_group)
main.command(wk)
