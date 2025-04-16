"""Entry-points for polarization-sensitive optical coherence tomography converter."""

from cyclopts import App

from linc_convert.cli import main, modalities_group

help = "Converters for PS-OCT .mat files"
psoct = App(name="psoct", help=help, group=modalities_group)
main.command(psoct)
