# aion-flow_v2

Data pipeline for *Probabilistic probes for galaxy evolution: signatures of AGN
feedback in the AION foundation model*. It builds the training inputs, the
label table and the split from public SRG/eROSITA-DE DR2, DESI DR1 and Legacy
Survey DR10 data.

Under construction; the plan is in `PLAN.md`.

```sh
uv venv && uv pip install -e ".[dev]"
make test
```
