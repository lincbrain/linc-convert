"""Entry-points for polarization-sensitive optical coherence tomography converter."""

from cyclopts import App

from linc_convert.cli import main

help = "Converters for PS-OCT .mat files"
psoct = App(name="psoct", help=help)
main.command(psoct)
