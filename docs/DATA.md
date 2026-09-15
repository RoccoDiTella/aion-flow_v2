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
| 6 | Legacy Survey DR10 cutouts | one per target | `ls-dr10`, 160 px at 0.262"/px, bands griz, centred on the DESI target position |

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
5. **Spectral class.** Rows whose DESI `SPECTYPE` is `STAR` are dropped
   (`spectype_not_stellar`). We neither train nor predict on stars, so they leave
   here rather than being carried through labels, spectra and cutouts to be
   filtered at the end; dropping them early also avoids fetching cutouts nothing
   reads. A Galactic star's redshift is real but is not a distance, so its
   luminosity would be meaningless, and no redshift-quality flag rejects it: a
   star sits at z of order 1e-5 with `ZWARN == 0`.
6. **Shared targets.** A target adopted by several detections: if every pair of
   X-ray positions is within 15", the group is a split source (all rows flagged
   `split_source`, excluded from the sample); otherwise a collision, and the row
   with the highest `NWAY_dist_post` (then `NWAY_p_any`, then the smallest
   separation) wins.
7. **Sample.** Rows with a fetched spectrum and a fetched cutout, minus
   split-source rows. Every sample target is unique.

Not sample cuts, carried instead: `DET_LIKE_0 > 6` (a label gate on the
broad-band heads; the Main catalogue's own inclusion threshold), redshift
quality (`has_z`) and WISE availability (`has_wise`).

Counts of the canonical run: 1,591,243 NWAY rows in, 2,203 stars dropped,
129,360 crossmatch rows out, 129,356 in the sample (the only further loss is two
split-source pairs), split 103,485 / 12,935 / 12,936. Full detail in
`data/provenance/crossmatch.json` and `manifest_split.json`.

## 3. Split

The sample sorted by `targetid`, permuted by `numpy.random.RandomState(42)`,
and cut at the cumulative fractions 0.8 / 0.9 (rounded to whole rows) into
train, val, test. The assignment depends only on the sample, the seed and the
fractions. Sizes: 103,485 train, 12,935 val, 12,936 test of 129,356.

## 4. Outputs

### `data/work/crossmatch.parquet`

One row per (detection, target). Columns: `targetid`; from NWAY `ero_detuid`,
`xray_ra`, `xray_dec` (the X-ray position), `ls10_ra`, `ls10_dec`,
`nway_p_any`, `nway_p_i`, `nway_threshold6`, `nway_dist_post`,
`ls10_flux_w1..w3`, `ls10_flux_ivar_w1..w3` (nanomaggies); from DESI
`target_ra`, `target_dec`, `survey`, `program`, `healpix`, `spectype`, `z`,
`zwarn`; `sep_arcsec` (target to LS10 position) and `split_source`.

### `data/work/labels.csv`

Every crossmatch column, plus, per band `b` in `1` (0.2-2.3 keV), `p2`
(0.5-1.0), `p3` (1.0-2.0):

| column | definition |
|---|---|
| `log_flux_<b>` | log10 `ML_FLUX`, NaN where the flux is not a measurement |
| `log_flux_<b>_sig_lo`, `_sig_hi` | split-normal errors in dex: `-log10(1 - LOWERR/F)`, `log10(1 + UPERR/F)`; the value is NaN if either exceeds 1.5 dex or the lower error swallows the flux |
| `det_like_0`, `det_like_<b>` | detection likelihood (`det_like_0` is the broad band's) |
| `ape_cts_<b>`, `ape_bkg_<b>`, `ape_exp_<b>` | the Poisson triple: aperture counts N (source plus background), background B, exposure t; N ~ Poisson(lambda t + B). A wrapped int16 count makes the triple missing; a negative background is clipped to 0 and flagged in `ape_bkg_negative_<b>` |

`log_lx` = `log_flux_1 + log10(4 pi D_L^2)` at Planck18, NaN below
`labels.z_floor` (0.001, about 4 Mpc). The floor is not a guard against dividing
by zero; it is the statement that the redshift is cosmological before it is used
as a distance. Stars are already gone by this point, so it is the safety net for
anything misclassified the other way, and it catches 154 rows of the canonical
run.

From CIGALE, one fit per target (same survey and program as the DESI
observation, then a main-survey fit, then lowest chi2): `logmstar_cigale`,
`log_sfr`, each with `_sig_lo` and `_sig_hi` equal to the catalogue error, NaN
where the fit failed (both values exactly 0), a -99 sentinel is present, the
quantity's PDF flag is outside (0.2, 5), the error is non-positive, or the
error exceeds 3 dex.

Label availability in the canonical run, of 129,360 crossmatch rows:
`log_flux_1` 129,294, `log_lx` 129,140, `log_flux_p2` 121,051, `log_flux_p3`
118,263, `logmstar_cigale` 116,318, `log_sfr` 104,050. The aperture triples are
complete for every row and every band. The CIGALE gates cost 11,651 masses to a
broad PDF and 22,246 star formation rates, of which 1,671 to the 3 dex error cap
alone (`labels.json`).

### `data/work/manifest.csv` and `split.csv`

Manifest, one row per crossmatch row: `targetid`, `ero_detuid`, `in_sample`,
`split` (blank outside the sample), `has_spectrum`, `has_z` (finite, `z > 0`,
`zwarn == 0`), `has_wise` (an LS10 band with flux > 0 and ivar > 0),
`has_image` (a cutout file), `spectype`, `z`, `zwarn`, `target_ra`,
`target_dec`, `ls10_flux_w1..w3`, `survey`, `program`, `healpix`,
`split_source`. Split: `targetid, split` for the sample.

Coverage within the sample of 129,356: `has_spectrum` and `has_image` 129,356
each, by construction; `has_wise` 129,343 (99.99%); `has_z` 126,392 (97.7%).

### `data/staged/{train,val,test}.h5`

Inputs only, rows in the order of `spectra/source.h5`.

| dataset | dtype | shape |
|---|---|---|
| `targetid` | int64 | (n,) |
| `spectra`, `spectra_ivar` | float32 | (n, 7781); grid 3600.0 + 0.8 k Angstrom, cameras B, R, Z coadded by inverse variance |
| `spectra_lambda` | float32 | (7781,) |
| `redshift`, `flux_w1`, `flux_w2`, `flux_w3` | float32 | (n,) |
| `image_flux` | float32 | (n, 4, 160, 160); griz |
| `has_z`, `has_wise` | bool | (n,) |

Attributes: `image_bands = [DES-G, DES-R, DES-I, DES-Z]`, `image_size = 160`,
`split`. Chunks are row-aligned (every axis but the first full), gzip level 4
on the spectra, none on the images.

### `data/work/line_features.csv`

One row per sample target: `targetid`, `split`, `spectype`, `z`, and for each
of `oiii_5007`, `nev_3426`, `halpha`, `hbeta`: `<line>_flux` (the line's
integrated flux, flux-density units times observed Angstrom, from a
single-component Gaussian fit over a local linear continuum; 0 where the line
is outside the coverage or the fit failed), `<line>_flux_err`, `<line>_an`
(amplitude over median noise), `<line>_status`. `line_fits.csv` holds every
fitted parameter.

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
