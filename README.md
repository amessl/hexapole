# Multi-objective Bayesian Optimization Pipeline

BoTorch-based pipeline for your 13-parameter / 3-objective (EIC intensity) process.

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
   appended to the main CSV with the next index numbers, and you can
   immediately refit + get the next batch in the same command.

## CLI usage

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

## Design choices — please review these

- **Parameter bounds** (`PARAM_BOUNDS` in the config section): inferred from
  the observed range in your initial 100 samples — `[-120, 0]` for the 12
  stage/output channels and `[0, 400]` for `RF`. If your hardware's true
  travel/power limits differ, edit `PARAM_BOUNDS` accordingly — this directly
  constrains where new candidates can be proposed.
- **Objective direction** (`MAXIMIZE_OBJECTIVE`): all three EIC channels
  default to **maximize**. If any of them is actually something you want to
  *suppress* (e.g. a background/contaminant signal), flip its entry to
  `False`.
- **Log-transform of objectives** (`LOG_TRANSFORM_OBJECTIVE`): your EIC values
  span ~6 orders of magnitude (1e-7 to ~0.6), which is very hard for a
  standard GP to fit directly (the huge dynamic range dominates the kernel
  lengthscale/noise fitting). All three objectives are log10-transformed
  before fitting by default; hypervolume/Pareto calculations are done
  consistently in that transformed space. Set an entry to `False` to disable.
- **Kernel**: `ScaleKernel(RBFKernel(ard_num_dims=d))` per objective, as
  requested — one lengthscale per input dimension (ARD), fit via
  `ExactMarginalLogLikelihood`.
- **Acquisition function**: `qLogExpectedHypervolumeImprovement`, not the
  vanilla `qExpectedHypervolumeImprovement`. BoTorch's own code emits a
  `NumericsWarning` recommending the log-space variant because the classic
  formulation is prone to vanishing gradients during acquisition
  optimization; it is a drop-in replacement with identical semantics.
- **Reference point**: computed automatically as the worst observed value per
  objective (in the maximize/log-transformed space) minus 10% of the observed
  range — a standard, data-driven heuristic. You can pass your own via
  `pipe._ref_point(...)` if you'd rather fix a physically meaningful
  worst-case point.
- **Batch optimization**: `optimize_acqf(..., sequential=True)` — greedily
  optimizes one candidate at a time conditioning on the ones already chosen
  in the batch, which is the standard/robust approach for `q>1` with EHVI-type
  acquisition functions.

This was tested end-to-end against your uploaded CSV (100 initial rows):
fitting, suggesting 5 candidates, simulating measurements, ingesting, and
refitting all ran successfully.
