# The model

The probe of the paper's Section 2.2 and Appendix B, the objective of its
Appendix C, and what this package does where the paper is silent. Written from
the paper; the tests quote the sentence each piece implements.

## 1. Layout

| module | what it is |
|---|---|
| `config.py` | the run recipes in `configs/`, and the optimizer, once |
| `data.py` | a staged split joined to its labels, the standardizers, the batch |
| `tokenize.py` | AION's frozen codecs, run once per split |
| `encoder.py` | the frozen backbone, the CLS read path, the readout MLPs |
| `flows.py` | the conditional spline head and the KDE prior |
| `poisson.py` | the count log-pmf, the node placement, the quadrature |
| `objective.py` | the 15 subsets, the sampler, the per-head likelihood |
| `train.py` | the loop, the selection, the checkpoint |
| `evaluate.py` | 15 combinations, information gain, R2, coverage |
| `baseline.py` | the emission-line flow |
| `analysis.py` | sSFR, hardness ratios, the within-object correlation, the bootstrap |
| `figures.py` | Figures 1 to 3 and Table 1 |
| `ablations.py` | Appendix B only: the pooling arms and the random-encoder control |

Stand-in encoders and codecs live in `tests/model/`, never in the package.

## 2. Order of operations

```
make all                                       # Phase 1, through line_features
make tokenize DEVICE=cuda                      # staged/tokens_{train,val,test}.h5
make train RUN=configs/marginals.yaml OUT=runs/marginals DEVICE=cuda
make train RUN=configs/rates.yaml     OUT=runs/rates     DEVICE=cuda
make train RUN=configs/joint4.yaml    OUT=runs/joint4    DEVICE=cuda
make baseline OUT=runs/baseline
make evaluate OUT=runs/marginals DEVICE=cuda   # and for each run
make evaluate OUT=runs/baseline BASELINE=1
make analysis DEVICE=cuda
python -m aionflow_model.figures --analysis results --out figures \
    --marginals runs/marginals --baseline runs/baseline
make ablations DEVICE=cuda                     # Appendix B, not one of the reported runs
```

Tokenizing is a step of its own because AION's codecs cost about half a second
per source, two orders of magnitude more than the encoder pass they feed, and
they are frozen and deterministic, so caching them is exact.

## 3. The probe

Frozen AION-1-B: 12 pre-norm blocks, width 768, 12 heads of 64, QK norm, gated
SiLU MLP, no projection biases. One extra token with its own residual stream,
initial state N(0, 0.02^2). At each block, with `X` the unmasked data tokens and
a hat the block's frozen pre-norm, the data path is untouched and the CLS reads
it through the same projections plus its three rank-64 deltas:

```
q = (W_Q + dQ) x_cls,  K = (W_K + dK) X,  V = (W_V + dV) X
Attn(q, K, V) = V softmax(K' q / sqrt(64))
```

QK norm acts after the deltas; the frozen output projection, residual and MLP
follow. The CLS is in neither K nor V. Each delta is `B A` with `A` in R^{64x768}
drawn N(0, 1/768) and `B` zero, so the read starts as the frozen attention. The
context is the readout MLP of the final CLS state after the encoder's output
LayerNorm.

Two consequences of writing nothing back. The data stream is AION's own
`Block.forward`, so it cannot diverge from the pretrained model. And nothing
trainable feeds it, so it carries no gradient: the data pass runs under
`no_grad` and only the CLS's own stream is kept for backward.

Trained parameters: 768 in the CLS, 3,538,944 in the deltas, 528,128 per readout,
and one flow per head (1,099,960 features=1, 1,151,344 features=2, 1,250,016
features=4). The three runs come to 11,731,536, 5,219,184 and 5,317,856.

## 4. The count likelihood

`N_b ~ Poisson(lambda_b t_b + B_b)`. A rate head is trained on the marginal
likelihood of the counts, by a uniform-weight Riemann sum on K = 12 equally
spaced closed nodes per axis spanning +-5 standardized units. Nodes are placed
per source and band by a Laplace proposal under the unit-scale standardized
prior, at centre `u/(1+s^2)` and scale `s/sqrt(1+s^2)` with
`s = sqrt(N)/(ln 10 . scale . (N-B))`, `N-B` floored at half a photon and `N` at
one. A dimension with no measurement falls back to the fixed prior grid with the
Jacobian carried exactly. Evaluation uses the same quadrature, and so does the
KDE prior the information gain is measured against.

Every dimension of a head therefore reaches the sum in one of three states:
pinned at an observed standardized value, integrated on a recentred grid against
a Poisson factor, or integrated on the fixed prior grid. A source with nothing
observed in a head is not scored by it, because integrating out every dimension
gives log 1 whatever the model says.

## 5. Appendix B

`ablations.py` is the only module the main path never imports, because the package
trains one architecture and these arms exist to say what is lost by reading the
encoder differently. It holds the two comparisons that test a claim:

- the **pooling comparison** - the CLS read against a bare attentive probe (one
  learned query, single-head cross-attention over the final tokens, a
  per-modality affine on the tokens and a presence embedding on the query) and a
  masked mean over the same tokens, all three sharing the readouts, flows, split,
  schedule, seed and sampled modality dropout, on the two heads flux and log LX;
- the **random-encoder control** - the same mean pool and heads on a frozen
  encoder of identical architecture that was never pretrained, with all four
  modalities always present.

Trained parameters come to 6,795,888, 5,625,456 and 3,256,176, against the
paper's 6.8M, 5.6M and 3.3M. That arithmetic is what pins "bare": a
feed-forward or a second layer overshoots, and dropping the output projection
lands at 5.0M.

The four-token and cosine-schedule grid of the same appendix is a hyperparameter
search rather than an architecture claim and is deliberately absent.

## 6. Choices the paper does not state

Each is written into the run directory, so a reader can see what was done.

| choice | what this package does |
|---|---|
| modality-dropout clamp | the size is drawn over the k modalities the source has, which is the paper's rule whenever k = 4 and never returns an empty set |
| rows a mixed joint trains on | at least one observed scalar, not merely one observed dimension. This is a deliberate departure from the appendix, which integrates any missing dimension out without qualification: a source with both rates and neither scalar can only tell the four-dimensional joint what the dedicated rate head already carries, and costs K^2 = 144 times the nodes to say it. On our sample that is 4.6% of sources and two thirds of the joint's whole quadrature budget. Training only; every test source still has posterior draws, so nothing is lost from the within-object correlation |
| training objective | the mean over heads of the per-row NLL, not the sum, so one set of learning rates serves a five-head run and a one-head run |
| batch chunking | a batch is scored in chunks; each head's mean is taken over its scorable rows in the whole batch, so the accumulated gradient is the whole batch's |
| validation metric | the unweighted mean over trained heads of their per-row NLL, on the rows each head can score |
| validation masks | one conditioning subset per source, drawn once under its own seed and kept, so the metric is comparable across epochs |
| common subsample | test sources with all four modalities and every one of the head's targets, fixed across the 15 rows; its size is an output |
| baseline context encoder | the readout's shape with `Linear(4, 512)` and no leading LayerNorm, since the four fluxes arrive standardized |
| bootstrap | 1,000 resamples of test sources, for Figure 1's errors and Figure 2's bands |
| Figure 2 smoothing | a centred rolling mean over a rank-ordered 8% window in redshift |
| rho | the Pearson correlation of the hardness ratio and log sSFR over a source's draws; "galaxies" is DESI spectype GALAXY |
| sSFR integral | a Riemann sum over the stellar-mass axis to eight standardized units, wider than the Poisson quadrature's five because truncating at five costs the tails of log sSFR |
| KDE prior | fitted on every complete training row, at Scott's bandwidth with the full covariance, matching `scipy.stats.gaussian_kde` |

## 7. Against the paper

Filled in after the three runs; the paper's numbers will move before submission,
so this table is a comparison, not a target.

| quantity | paper | here |
|---|---|---|
| IG, four modalities, X-ray flux | 0.299 | *pending* |
| IG, four modalities, log LX | 1.252 | *pending* |
| IG, four modalities, log SFR | 0.937 | *pending* |
| IG, four modalities, log M* | 0.957 | *pending* |
| baseline IG, X-ray flux | 0.197 | *pending* |
| R2, four modalities, X-ray flux | 0.496 | *pending* |
| redshift alone, log LX | 1.02 | *pending* |
| sSFR under the joint over independent heads | 0.24 nats | *pending* |
| counts under the rate joint over its prior | 0.957 nats | *pending* |
| rho negative among test galaxies | 74% | *pending* |
| rho negative below z = 0.7 | 85% | *pending* |
| 68 / 90 / 95% coverage | 66.2-68.0 / 88.5-89.7 / 93.9-94.8% | *pending* |
| trained parameters, the three runs | 11.7M / 5.2M / 5.3M | 11,731,536 / 5,219,184 / 5,317,856 |
| trained parameters, Appendix B's arms | 6.8M / 5.6M / 3.3M | 6,795,888 / 5,625,456 / 3,256,176 |
