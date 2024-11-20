"""Entry-points for Webknossos annotation converter."""

from cyclopts import App

from linc_convert.cli import main

help = "Converters for Webknossos annotation"
wk = App(name="wk", help=help)
main.command(wk)
