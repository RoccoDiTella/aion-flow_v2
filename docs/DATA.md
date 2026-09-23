# Data

This page describes how the sample is built: what each step reads, the rules it
applies and what it writes. All counts are from our run and match the ledgers in
`data/provenance/`. Constants are in `config.yaml`.

## Inputs

| # | file | rows | contents |
|---|---|---|---|
| 1 | `eRASSc3_Main_LS10_Public_27Jul2026.fits.gz` | 1,591,243 | SRG/eROSITA-DE DR2 (eRASS:3) NWAY counterparts to Legacy Survey DR10, one row per candidate |
| 2 | `eRASS3_Main_v1.3.fits` | 1,975,540 | eRASS:3 Main catalogue, one row per X-ray detection |
| 3 | `zall-pix-iron.fits` | 28,425,963 | DESI DR1 redshift catalogue, one row per coadd |
| 4 | `IronPhysProp_v1.2.fits` | 17,149,172 | DESI DR1 CIGALE fits to grz and W1–W4 photometry at the DESI redshift |
| 5 | DESI DR1 healpix coadds | — | spectra from cameras B, R and Z, read by HTTP range for our targets only |
| 6 | Legacy Survey DR10 cutouts | one per target | griz, 160 px at 0.262″/px, centred on the DESI target |

## Selection

`crossmatch` applies rules 1 to 6 and `manifest_split` rule 7.

1. **DESI targets.** We keep `ZCAT_PRIMARY` rows with `TARGETID > 0` and a finite
   position. There is no redshift-quality cut.
2. **NWAY counterparts.** We keep rows with `NWAY_match_flag == 1`, after
   collapsing exact duplicates. A detection with several such rows keeps the one
   with the highest `NWAY_p_i`, with ties broken by `NWAY_dist_post`.
3. **Association.** We take the DESI targets within 1″ of the counterpart's
   `LS10_RA, LS10_DEC`. Where there are several, we prefer a main-survey Legacy
   Survey release (9010, 9011 or 9012, in bits 42–57 of the TARGETID), then the
   nearest. The two catalogues index different Legacy Survey releases (DR9 and
   DR10), so they cannot be joined on an identifier.
4. **Reliability.** We require `NWAY_p_any > NWAY_threshold6`, NWAY's own
   per-tile threshold, or `NWAY_p_any >= 0.05` where that calibration is missing.
5. **Stars.** We drop targets DESI classifies as `STAR`, since we neither train nor
   predict on them. A Galactic star has a real redshift of order 10⁻⁵ with
   `ZWARN == 0`, so no redshift-quality flag would remove it.
6. **Shared targets.** When several detections adopt the same target and every pair
   of X-ray positions is within 15″, the group is a split source: its rows are
   flagged `split_source` and left out of the sample in rule 7. Otherwise the row
   with the highest `NWAY_dist_post` is kept (then `NWAY_p_any`, then the smallest
   separation).
7. **Sample.** Every target with both a spectrum and a cutout that is not a split
   source.

Detection likelihood (`DET_LIKE_0 > 6`), redshift quality and WISE availability
are not sample cuts. They are carried as label gates and presence flags.

Of 1,591,243 NWAY rows, 2,203 stars are dropped and 129,360 crossmatch rows remain.
The sample has 129,356 sources; the only further loss is two split sources.

## Split

We sort the sample by `targetid`, permute it with `numpy.random.RandomState(42)`
and cut it at 80/10/10, giving 103,485 training, 12,935 validation and 12,936 test
sources. The split depends only on the sample, the seed and the fractions.

## Outputs

### `data/work/crossmatch.parquet`

One row per (detection, target). From NWAY: `ero_detuid`, `xray_ra`, `xray_dec`,
`ls10_ra`, `ls10_dec`, `nway_p_any`, `nway_p_i`, `nway_threshold6`,
`nway_dist_post`, `ls10_flux_w1..w3` and `ls10_flux_ivar_w1..w3` (nanomaggies).
From DESI: `targetid`, `target_ra`, `target_dec`, `survey`, `program`, `healpix`,
`spectype`, `z`, `zwarn`. Also `sep_arcsec` (target to LS10 position) and
`split_source`.

### `data/work/labels.csv`

Every crossmatch column, plus the following for each band `b`: `1` (0.2–2.3 keV),
`p2` (0.5–1.0 keV) and `p3` (1.0–2.0 keV).

| column | definition |
|---|---|
| `log_flux_<b>` | log10 of `ML_FLUX`; NaN where the flux is not a measurement |
| `log_flux_<b>_sig_lo`, `_sig_hi` | split-normal errors in dex, `-log10(1 - LOWERR/F)` and `log10(1 + UPERR/F)`; the flux is NaN if either exceeds 1.5 dex or the lower error exceeds the flux |
| `det_like_0`, `det_like_<b>` | detection likelihood; `det_like_0` is the broad band's |
| `ape_cts_<b>`, `ape_bkg_<b>`, `ape_exp_<b>` | aperture counts N, background B and exposure t, with N ~ Poisson(λt + B). A wrapped int16 count makes the triple missing; a negative background is set to 0 and flagged in `ape_bkg_negative_<b>` |

`log_lx` is `log_flux_1 + log10(4π D_L²)` for a Planck18 cosmology. It is NaN below
z = 0.001 (about 4 Mpc), where a redshift is not a reliable distance. This affects
154 rows.

`logmstar_cigale` and `log_sfr` come from one CIGALE fit per target, chosen by
matching survey and program, then a main-survey fit, then the lowest χ². Each has
`_sig_lo` and `_sig_hi` equal to the catalogue error. A value is NaN where the fit
failed (both values exactly 0), holds a −99 sentinel, has a PDF flag outside
(0.2, 5), or has an error that is non-positive or larger than 3 dex.

Labels available among the 129,360 crossmatch rows:

| label | rows |
|---|---|
| `log_flux_1` | 129,294 |
| `log_lx` | 129,140 |
| `log_flux_p2` | 121,051 |
| `log_flux_p3` | 118,263 |
| `logmstar_cigale` | 116,318 |
| `log_sfr` | 104,050 |

The aperture triples are complete for every row and band.

### `data/work/manifest.csv` and `split.csv`

The manifest has one row per crossmatch row: `targetid`, `ero_detuid`,
`in_sample`, `split` (blank outside the sample), `has_spectrum`, `has_z` (finite
z > 0 with `zwarn == 0`), `has_wise` (an LS10 WISE band with positive flux and
inverse variance), `has_image`, `spectype`, `z`, `zwarn`, `target_ra`,
`target_dec`, `ls10_flux_w1..w3`, `survey`, `program`, `healpix` and
`split_source`. `split.csv` holds `targetid, split` for the sample.

Within the sample, every source has a spectrum and an image, 129,343 (99.99%) have
WISE and 126,392 (97.7%) have a good redshift.

### `data/staged/{train,val,test}.h5`

Model inputs only; labels stay in `labels.csv`.

| dataset | dtype | shape |
|---|---|---|
| `targetid` | int64 | (n,) |
| `spectra`, `spectra_ivar` | float32 | (n, 7781), on a grid from 3600 Å in steps of 0.8 Å, with B, R and Z coadded by inverse variance |
| `spectra_lambda` | float32 | (7781,) |
| `redshift`, `flux_w1`, `flux_w2`, `flux_w3` | float32 | (n,) |
| `image_flux` | float32 | (n, 4, 160, 160), griz |
| `has_z`, `has_wise` | bool | (n,) |

### `data/work/line_features.csv`

The emission-line baseline's inputs, one row per source: `targetid`, `split`,
`spectype`, `z`, and for each of `oiii_5007`, `nev_3426`, `halpha` and `hbeta` the
columns `<line>_flux`, `<line>_flux_err`, `<line>_an` (amplitude over median
noise) and `<line>_status`. Fluxes come from a single Gaussian over a local linear
continuum, fitted on our own spectra, and are 0 where the line falls outside the
coverage or the fit failed. `line_fits.csv` holds every fitted parameter.

## Ledgers

Each step writes `data/provenance/<step>.json` with the config and its sha256,
each input's path, size and sha256, the counts in and out, and an ordered list of
filters with the rows each kept and dropped. `validate.json` lists every check and
its result. `make validate` must pass before the staged files are used.
