# Data contract

What each step reads, what it writes, and the rules it applies. Counts marked
*pending* are filled from `data/provenance/` once the canonical run has been
made. Constants are in `config.yaml`; this page states them where they matter.

## 1. Inputs

| # | file | rows | what it is |
|---|---|---|---|
| 1 | `eRASSc3_Main_LS10_Public_27Jul2026.fits.gz` | 1,591,243 | SRG/eROSITA-DE DR2 (eRASS:3) NWAY counterparts to Legacy Survey DR10; one row per candidate, `NWAY_match_flag` 1 for the adopted counterpart |
| 2 | `eRASS3_Main_v1.3.fits` | 1,975,540 | eRASS:3 Main catalogue, one row per X-ray detection |
| 3 | `zall-pix-iron.fits` | 28,425,963 | DESI DR1 redshift catalogue, one row per (target, survey, program) coadd; `ZCAT_PRIMARY` marks DESI's best observation |
| 4 | `IronPhysProp_v1.2.fits` | 17,149,172 | DESI DR1 CIGALE fits to grz + W1-W4 photometry at the DESI redshift |
| 5 | DESI DR1 healpix coadds | read by HTTP range | `coadd-{survey}-{program}-{healpix}.fits`, cameras B, R, Z |
| 6 | Legacy Survey DR10 cutouts | one per target | `ls-dr10`, 160 px at 0.262"/px, bands griz, centred on the DESI fibre position (`cutouts.position`, or `target` for the catalogue position; a fraction of a pixel apart) |

Sizes, URLs and checksums: `config.yaml`. The fetch step (`data/provenance/raw.json`)
records what was retrieved and when.

## 2. Selection

Applied in `crossmatch` (rules 1 to 5) and `manifest_split` (rule 6).

1. **DESI targets.** `ZCAT_PRIMARY`, `TARGETID > 0`, finite position. No
   redshift-quality cut.
2. **NWAY rows.** Exact duplicates collapsed (first kept). `NWAY_match_flag == 1`.
   A DETUID with several primary rows keeps the highest `NWAY_p_i`, ties by
   `NWAY_dist_post`.
3. **Association.** All DESI targets within 1.0" of `LS10_RA, LS10_DEC`; prefer a
   TARGETID whose release bits (42 to 57) are a main-survey Legacy Survey
   release (9010, 9011, 9012), then the nearest. No identity join is possible:
   DESI DR1 targeting indexes LS DR9, the NWAY table indexes DR10.
4. **Reliability.** `NWAY_p_any > NWAY_threshold6` where the calibration is
   present; `NWAY_p_any >= 0.05` where it is absent. Both branch counts are in
   the ledger.
5. **Shared targets.** A target adopted by several detections: if every pair of
   X-ray positions is within 15", the group is a split source (all rows flagged
   `split_source`, excluded from the sample); otherwise a collision, and the row
   with the highest `NWAY_dist_post` (then `NWAY_p_any`, then the smallest
   separation) wins.
6. **Sample.** Rows with a fetched spectrum, minus split-source rows.

Not sample cuts, carried instead: `DET_LIKE_0 > 6` (a label gate on the
broad-band heads; the Main catalogue's own inclusion threshold), redshift
quality (`has_z`), WISE and image availability (`has_wise`, `has_image`).

Counts of the canonical run: *pending* (`data/provenance/crossmatch.json`,
`manifest_split.json`).

## 3. Split

Connected components of the bipartite graph with detections and targets as
nodes and sample rows as edges, keyed by the smallest DETUID in the component.
Each key is hashed with keyed blake2b (8-byte digest, salt in `config.yaml`),
mapped to [0, 1), and cut at 0.8 and 0.9 for train, val, test. The assignment
depends only on (key, salt, fractions). Row fractions are checked against a
tolerance. Sizes: *pending*.

## 4. Outputs

### `data/work/crossmatch.parquet`

One row per (detection, target). Columns: `targetid`; from NWAY `ero_detuid`,
`xray_ra`, `xray_dec` (the X-ray position), `ls10_ra`, `ls10_dec`,
`ls10_release`, `ls10_brickid`, `ls10_objid`, `nway_p_any`, `nway_p_i`,
`nway_p_single`, `nway_match_flag`, `nway_threshold6`, `nway_dist_post`,
`nway_dist_bayesfactor`, `nway_sep_arcsec`, `ls10_flux_w1..w3`,
`ls10_flux_ivar_w1..w3`, `ls10_shape_r`, `ls10_sersic`, `ls10_type`,
`ls10_xray_proba`, `exgal_prob_starex`, `class_gal_exgal`,
`simbad_known_galactic`; from DESI `target_ra`, `target_dec`, `fiber_ra`,
`fiber_dec`, `survey`, `program`, `healpix`, `spectype`, `z`, `zwarn`,
`deltachi2`; match diagnostics
`sep_arcsec`, `n_candidates`, `preferred_over_nearest`, `desi_release`,
`is_main_survey`, `reliability_branch`, `split_source`, `collision_group_size`.

### `data/work/labels.csv`

Every crossmatch column, plus, per band `b` in `1` (0.2-2.3 keV), `p1`
(0.2-0.5), `p2` (0.5-1.0), `p3` (1.0-2.0), `p4` (2.0-5.0):

| column | definition |
|---|---|
| `log_flux_<b>` | log10 `ML_FLUX`, NaN where the flux is not a measurement |
| `log_flux_<b>_sig_lo`, `_sig_hi` | split-normal errors in dex: `-log10(1 - LOWERR/F)`, `log10(1 + UPERR/F)`; the value is NaN if either exceeds 1.5 dex or the lower error swallows the flux |
| `det_like_0`, `det_like_<b>` | detection likelihood |
| `ape_cts_<b>`, `ape_bkg_<b>`, `ape_exp_<b>` | the Poisson triple: aperture counts N (source plus background), background B, exposure t; N ~ Poisson(lambda t + B). A wrapped int16 count makes the triple missing; a negative background is clipped to 0 and flagged in `ape_bkg_negative_<b>` |
| `ape_radius_<b>`, `ape_pois_<b>`, `ml_cts_<b>`, `ml_rate_<b>`, `ml_exp_<b>`, `ml_eef_<b>` | carried raw |

Broad-band aliases the trainer reads: `log_ml_flux_1`, `flux_sig_lo`,
`flux_sig_hi`. `log_lx` = `log_flux_1 + log10(4 pi D_L^2)` at Planck18, NaN at
`z <= 0`.

From CIGALE, one fit per target (same survey and program as the DESI
observation, then a main-survey fit, then lowest chi2): `logmstar_cigale`,
`log_sfr`, each with `_sig_lo` and `_sig_hi` equal to the catalogue error, NaN
where the fit failed (both values exactly 0), a -99 sentinel is present, the
quantity's PDF flag is outside (0.2, 5), the error is non-positive, or the
error exceeds 3 dex; `ref_log_ssfr` = `log_sfr - logmstar_cigale` where both
survive, with `ref_log_ssfr_sig_indep`; `cigale_agnfrac`, `cigale_agnlum`,
`cigale_chi2`, `cigale_flag_masspdf`, `cigale_flag_sfrpdf`, `cigale_spectype`,
`cigale_survey`, `cigale_program`.

Label availability in the canonical run: *pending* (`labels.json`).

### `data/work/manifest.csv` and `split.csv`

Manifest, one row per crossmatch row: `targetid`, `ero_detuid`, `source_row`
(row in `spectra/source.h5`, -1 if none), `in_sample`, `split` (blank outside
the sample), `component`, `has_spectrum`, `has_z` (finite, `z > 0`,
`zwarn == 0`), `has_w1..w3` (LS10 flux > 0 and ivar > 0), `has_wise` (any
band), `has_image` (cutout present and readable), `spectype`, `z`, `zwarn`,
`target_ra`, `target_dec`, `flux_w1..w3` (LS10, nanomaggy), `survey`,
`program`, `healpix`, `split_source`. Split: `targetid, split` for the sample.

Coverage within the sample: *pending*.

### `data/staged/desi_{train,val,test}.hdf5`

Inputs only, rows sorted by `source_row`.

| dataset | dtype | shape |
|---|---|---|
| `source_row`, `desi_targetid` | int64 | (n,) |
| `spectra`, `spectra_ivar` | float32 | (n, 7781); grid 3600.0 + 0.8 k Angstrom, cameras B, R, Z coadded by inverse variance |
| `spectra_lambda` | float32 | (7781,) |
| `redshift`, `flux_w1`, `flux_w2`, `flux_w3`, `target_ra`, `target_dec` | float32 | (n,) |
| `image_flux` | float32 | (n, 4, 160, 160); griz, zero frame where `has_image` is false |
| `has_spectrum`, `has_z`, `has_wise`, `has_image` | bool | (n,) |

Attributes: `image_bands = [DES-G, DES-R, DES-I, DES-Z]`, `image_size = 160`,
`split`. Chunks are row-aligned (every axis but the first full), gzip level 4
on the spectra, none on the images. `summary.json` records rows and image
counts per split.

### `data/work/line_features.csv`

One row per sample target: `targetid`, `split`, `spectype`, `z`, and for each
of `oiii_5007`, `nev_3426`, `halpha`, `hbeta`: `<line>_flux` (observed-frame
integrated flux from a single-component fit; 0 where the line is outside the
coverage or the fit failed), `<line>_flux_err`, `<line>_an` (amplitude over
median noise), `<line>_status`. `line_fits.csv` holds every fitted parameter.
The Balmer decrement by class is in `line_features.json`.

## 5. Ledgers

`data/provenance/<step>.json`, one per step: the config path and its sha256,
each input's path, size and sha256, counts in and out, an ordered filter
ledger (`filter`, `kept`, `dropped`), and step-specific extras. `validate.json`
lists every check with its verdict.

## 6. Reproducing

```sh
uv venv && uv pip install -e ".[dev]"
make all
```

Steps run in order and each can be rerun alone with `make <step>`. The
`spectra` and `cutouts` steps resume; the others recompute. `make validate`
must pass before anything downstream reads the staged files.
