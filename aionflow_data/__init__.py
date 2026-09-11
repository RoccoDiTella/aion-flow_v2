"""Data pipeline for the AION X-ray paper.

Steps, in order: fetch_catalogs, crossmatch, labels, fetch_spectra,
fetch_cutouts, manifest_split, stage, validate, line_features.
"""

__version__ = "0.1.0"
