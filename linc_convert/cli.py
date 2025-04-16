"""Root command line entry point."""

from cyclopts import App, Group

help = "Collection of conversion scripts for LINC datasets"
main = App("linc-convert", help=help)


modalities_group = Group.create_ordered("Modality-specific converters")
plain_group = Group.create_ordered("Plain converters")
