# Multi-objective Bayesian Optimization Pipeline

BoTorch-based pipeline for 13-parameter ion transfer settings / 3-objective (EIC intensity) optimization process.

## Install

```bash
pip install -r requirements.txt
```

## Files

- `bo_pipeline.py` — the pipeline (importable as a module, or runnable as a CLI).
- Your CSV (e.g. `data.csv`) — main dataset, one row per measured sample.

## How it works

1. **Load** — `MultiObjectiveBOPipeline(csv_path=...)` reads all fully-measured
   rows from the CSV.
2. **Fit** — one `SingleTaskGP` per objective (RBF kernel, `ARD` per input dim),
   combined into a `ModelListGP`.
3. **Acquire** — `qLogExpectedHypervolumeImprovement` (the numerically robust,
   BoTorch-recommended log-space version of qEHVI) is optimized jointly for a
   batch of `n` candidates via `optimize_acqf(..., sequential=True)`.
4. **Suggest** — candidates are written to `<name>_candidates_to_measure.csv`
   with the objective columns left empty.
5. **Measure & ingest** — after running the experiment, fill in the empty
   objective columns in that candidates file and run `ingest`. The rows are
   appended to the main CSV with the next index numbers.

## CLI usage

**Multi-objective (default — all 3 EIC channels, qLogEHVI):**

```bash
# 1. Get the first batch of 5 candidates from your existing data
python bo_pipeline.py --csv data.csv suggest --n 5
# -> writes data_candidates_to_measure.csv

# 2. Run the experiment, fill in the 3 EIC columns in that file, then:
python bo_pipeline.py --csv data.csv ingest --file data_candidates_to_measure.csv --suggest-next 5
# -> appends the measured rows to data.csv, refits, writes the next batch

# Inspect current Pareto-optimal measured points at any time
python bo_pipeline.py --csv data.csv pareto
```

**Single-objective (one EIC channel, qLogEI):**

```bash
python bo_pipeline.py --csv data.csv --mode single --objective "EIC 2,9584 - 3,2623" suggest --n 5
python bo_pipeline.py --csv data.csv --mode single --objective "EIC 2,9584 - 3,2623" \
    ingest --file data_candidates_to_measure.csv --suggest-next 5
python bo_pipeline.py --csv data.csv --mode single --objective "EIC 2,9584 - 3,2623" best
```

Add `--minimize` after `--mode single --objective ...` to minimize
that channel instead of maximizing it. Single-objective mode shares the same
CSV/schema as multi-objective mode. The other two EIC columns may be left
blank on ingest if only the one to be optimized was measured. Rows with
missing objectives are simply skipped when you switch back to `--mode multi`
(which requires all three to be filled in).

## Programmatic usage

```python
from bo_pipeline import MultiObjectiveBOPipeline

pipe = MultiObjectiveBOPipeline(csv_path="data.csv")
pipe.fit()
candidates = pipe.suggest(n=5)          # DataFrame, param columns only (objectives = NaN)
pipe.write_candidates(candidates, "next_batch.csv")

# ... run experiment, fill in EIC columns in next_batch.csv ...

pipe.ingest("next_batch.csv")           # appends to data.csv
next_candidates = pipe.refit_and_suggest(n=5)
```

## Additional Info

- **Parameter bounds** (`PARAM_BOUNDS` in the config section): inferred from
  the observed range in the initial 100 samples — `[-120, 0]` for the 12
  stage/output channels and `[0, 400]` for `RF`.
- **Objective direction** (`MAXIMIZE_OBJECTIVE`): all three EIC channels
  default to **maximize**. If any of them is something to be
  *suppressed* (e.g. a background/contaminant signal), change entry to
  `False`.
- **Log-transform of objectives** (`LOG_TRANSFORM_OBJECTIVE`): EIC values
  span 6 orders of magnitude (1e-7 to ~0.6) which is very hard for a
  standard GP to fit directly (the huge dynamic range dominates the kernel
  lenght scale/noise fitting). All three objectives are log10-transformed
  before fitting by default and hypervolume/Pareto calculations are done
  consistently in that transformed space. Set entry to `False` to disable.
- **Kernel**: `ScaleKernel(RBFKernel(ard_num_dims=d))` per objective, as
  requested — one lengthscale per input dimension (ARD), fit via
  `ExactMarginalLogLikelihood`.
- **Acquisition function**: `qLogExpectedHypervolumeImprovement` in multi-
  objective mode (`--mode multi`, default), `qLogExpectedImprovement` in
  single-objective mode (`--mode single --objective "..."`). Both are the
  numerically robust, BoTorch-recommended log-space variants — the classic
  `qExpectedHypervolumeImprovement`/`qExpectedImprovement` formulations emit
  a `NumericsWarning` recommending these as drop-in replacements because they
  are prone to vanishing gradients during acquisition optimization.
- **Reference point**: computed automatically as the worst observed value per
  objective (in the maximize/log-transformed space) minus 10% of the observed
  range.
- **Batch optimization**: `optimize_acqf(..., sequential=True)` — greedily
  optimizes one candidate at a time conditioning on the ones already chosen
  in the batch, which is supposed to be the standard approach for `q>1` with EHVI-type
  acquisition functions.
