# Model

This page describes the probe (Section 2.2 and Appendix B of the paper), how it is
trained and scored (Appendix C), and the choices the paper leaves open. Where the
paper specifies something, the module's docstring quotes it.

## Modules

| module | contents |
|---|---|
| `config.py` | the run recipes in `configs/` and the optimizer settings |
| `data.py` | a staged split joined to its labels, the standardizers, batching |
| `tokenize.py` | AION's frozen codecs, run once per split |
| `encoder.py` | the frozen encoder, the CLS read path and the readout MLPs |
| `flows.py` | the conditional spline flow and the KDE prior |
| `poisson.py` | the count likelihood and its quadrature |
| `objective.py` | modality dropout and the per-head likelihood |
| `train.py` | the training loop, model selection and checkpoints |
| `evaluate.py` | information gain, R² and coverage over the 15 input combinations |
| `baseline.py` | the emission-line baseline |
| `cleaning.py` | which galaxies' X-rays star formation could explain |
| `analysis.py` | sSFR, hardness ratios and the within-object correlation |
| `figures.py` | Figures 1 to 3 and Table 1 |
| `ablations.py` | comparison arms, never imported by the rest of the package |

The tests run against a small stand-in encoder and codecs in `tests/model/`.

## The probe

The backbone is the frozen encoder of AION-1-B: 12 pre-norm transformer blocks of
width 768, 12 attention heads of 64 dimensions, QK normalization, a gated SiLU MLP
and no projection biases, 314M parameters in all. We add one CLS token with its own
residual stream, initialized from N(0, 0.02²).

At each block the CLS attends to the data tokens X through the block's own
projections, each with a rank-64 update:

```
q = (W_Q + ΔQ) x_cls,   K = (W_K + ΔK) X,   V = (W_V + ΔV) X
Attn(q, K, V) = V softmax(Kᵀq / √64)
```

QK normalization is applied after the updates, followed by the block's frozen
output projection, residual and MLP. Each update is ΔW = BA, with A drawn from
N(0, 1/768) and B set to zero, so training starts from the pretrained attention.
The data tokens never attend to the CLS, so their forward pass is exactly the
pretrained model's and needs no gradient.

After the last block, the CLS state passes through the encoder's output LayerNorm
and one readout MLP per head (768 → 512 → 256, with LayerNorms and dropout 0.05),
giving a 256-dimensional context.

## Heads and runs

Every head is a conditional neural spline flow on the context: 8 transforms, each
with 8 rational-quadratic bins on [−5, 5] and a masked autoregressive conditioner
with two hidden layers of 256. A head over several targets is a joint. Targets are
standardized with the mean and scale of the training split.

| run | heads | trained parameters |
|---|---|---|
| `marginals` | flux, log LX, log SFR, log M⋆, and (SFR, M⋆) | 11,731,536 |
| `rates` | (λ_P2, λ_P3) | 5,219,184 |
| `joint4` | (λ_P2, λ_P3, SFR, M⋆) | 5,317,856 |

Of these, 768 are the CLS token, 3,538,944 the rank-64 updates and 528,128 each
readout MLP. A flow has 1,099,960 parameters for one target, 1,151,344 for two and
1,250,016 for four.

## Modality dropout

At every training step each source is conditioned on a random subset of its
modalities. We draw the subset size uniformly from 1 to k, where k is the number of
modalities the source has, and then a subset of that size uniformly. One model
therefore covers all 15 combinations of S, I, W and Z.

## The count likelihood

X-ray counts follow N_b ~ Poisson(λ_b t_b + B_b), with exposure t_b and expected
background B_b. A rate head places its density on the latent rate λ_b, in
standardized log10 units, and is trained on the marginal likelihood of the counts
(Eq. 1 of the paper). The integral is a Riemann sum on K equally spaced nodes per
axis spanning ±5 standardized units.

For a measured band, the nodes are centred and scaled per source by a Laplace
approximation, at centre û/(1 + σ²) and scale σ/√(1 + σ²), where û is the
standardized plug-in rate and σ = √N / (ln 10 · s · (N − B)) for standardization
scale s. N − B is floored at half a photon and N at one. A band with no measurement,
or a missing dimension of a joint, uses the fixed grid instead.

Each dimension of a head is therefore either pinned at its observed value,
integrated against the Poisson factor on a recentred grid, or integrated on the
fixed grid. A source with no observed dimension in a head is not scored by it.

**Number of nodes.** Training uses K = 12. That is not enough to score heads with
latent rates. Rescoring one checkpoint at K = 12, 24 and 48 with all four
modalities, the information gain of the rate joint is 0.95, 0.06 and 0.35 nats,
and that of `joint4` 2.12, 1.44 and 1.67. The sequence alternates and shrinks, and
K = 48 is within about 0.06 nats of its limit. Score `rates` and `joint4` with
`NODES=48`. A dimension pinned at an observed value involves no quadrature, so the
four scalar heads are unaffected.

## Training

We use AdamW with (β₁, β₂) = (0.95, 0.999) and constant learning rates: 3×10⁻⁴ for
the readouts and the CLS, 10⁻³ for the flows and 3×10⁻⁵ for the rank-64 updates.
Weight decay is 10⁻⁴ on the readouts and flows, 0.1 on the updates and 0 on the CLS.
Batches hold 896 sources and gradients are clipped at norm 5. Training runs for at
most 40 epochs and stops after 5 without improvement in the validation loss. The
seed is 42 throughout.

The loss is the mean over heads of each head's mean negative log-likelihood per
source. A batch is processed in chunks of `CHUNK` sources, with each head's mean
taken over the whole batch, so the gradient does not depend on the chunk size.

`joint4` uses half these rates (`lr_scale: 0.5` in its recipe). At the full rates
its training loss rose after epoch 6.

## Star-formation cut

X-ray binaries and hot gas in star-forming galaxies emit X-rays roughly in
proportion to stellar mass and star formation rate. Where that emission could
account for a galaxy's observed X-rays, its X-ray labels do not measure the AGN.

We predict this luminosity from the CIGALE M⋆ and SFR with the relation of
Lehmer et al. (2016, Table 3), L_X = α₀(1 + z)^γ M⋆ + β₀(1 + z)^δ SFR, in both its
0.5–2 keV and 2–10 keV forms, each converted to 0.2–2.3 keV assuming a Γ = 2 power
law. A galaxy is flagged if its observed L_X is at most 10 times either prediction,
so that star formation could supply at least a tenth of its X-rays. Only DESI GALAXY
sources are flagged, since a quasar's light inflates its CIGALE SFR, and sources
without M⋆, SFR or z are kept.

With `exclude_sf_dominated: true`, a run neither trains nor validates on flagged
galaxies. The split itself is unchanged, and every test source is still scored. Of
the galaxies with CIGALE values, the cut flags 4.3% (6.5% at z < 0.7). Only `joint4`
uses it.

## Evaluation

`evaluate` scores the test split under all 15 input combinations. The information
gain is the mean over test sources of log p(y | inputs) − log p_KDE(y) in nats,
where p_KDE is a Gaussian KDE fitted on the training split. For rate heads both
terms are the counts likelihood above. R² uses the posterior mean in natural units,
and coverage is reported at 68, 90 and 95%.

`analysis` reads the joint posteriors, with 32,768 draws per source. It compares
the catalogue sSFR under the (SFR, M⋆) joint and under the independent heads,
builds hardness-ratio posteriors HR_λ = (λ_P3 − λ_P2)/(λ_P3 + λ_P2) from the rate
joint, and computes the within-object correlation ρ between HR_λ and sSFR from
`joint4`.

## Comparison arms

`ablations.py` holds three comparisons.

- **Pooling (Appendix B).** The CLS read against an attentive probe (one learned
  query with single-head cross-attention over the final tokens) and a masked mean
  pool, on the flux and log LX heads. They train 6,795,888, 5,625,456 and 3,256,176
  parameters.
- **Random encoder (Appendix B).** The mean pool on a randomly initialized encoder
  of the same architecture, with all four modalities present.
- **Finetuning.** The `marginals` run with the encoder unfrozen, as an upper bound
  on what the frozen representation leaves out. It trains 325,987,152 parameters.

`make ablations` runs the first two. The finetuning arm needs a smaller chunk to fit
on an H200:

```sh
python -m aionflow_model.ablations --arm finetune --out runs/finetune \
    --lr-backbone 3e-5 --chunk 256 --device cuda
```

## Choices the paper leaves open

Each is recorded in the run's `choices.json`.

| choice | what we do |
|---|---|
| modality-dropout size | drawn over the modalities a source has, so a subset is never empty |
| rows a mixed joint trains on | a head that mixes rates and scalars trains only on sources with at least one observed scalar. This removes 4.6% of training sources from `joint4`. The paper integrates any missing dimension out instead |
| loss over heads | the mean rather than the sum, so one set of learning rates serves every run |
| validation metric | the unweighted mean over heads of the per-source negative log-likelihood |
| validation subsets | one modality subset per source, drawn once and kept for every epoch |
| common subsample | test sources with all four modalities and all of a head's targets, fixed across the 15 combinations |
| baseline encoder | the readout MLP with a 4-dimensional input and no leading LayerNorm |
| bootstrap | 1,000 resamples of the test sources |
| Figure 2 smoothing | a rolling mean over 8% of the test sources, ordered by redshift |
| ρ | the Pearson correlation of HR_λ and log sSFR over a source's draws; galaxies are DESI `SPECTYPE` GALAXY |
| sSFR integral | a Riemann sum over stellar mass to ±8 standardized units |
| KDE prior | full covariance at Scott's bandwidth, fitted on complete training rows |
