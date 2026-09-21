"""
Multi-objective Bayesian optimization pipeline (BoTorch) for a 13-parameter /
3-objective (EIC intensity) experimental process.

Workflow
--------
1. `MultiObjectiveBOPipeline(csv_path=...)` loads all measured samples from the CSV.
2. `.fit()` fits one GP (RBF kernel) per objective, wrapped in a `ModelListGP`.
3. `.suggest(n=5)` optimizes q-Expected-Hypervolume-Improvement (qEHVI) for a
   batch of `n` new parameter sets and writes them to a "candidates" CSV with
   empty objective columns for you to fill in after running the experiment.
4. After measuring, fill in the objective columns of that candidates CSV and
   call `.ingest(candidates_csv)`. This appends the new rows (with real index
   numbers) to the main CSV, refits the GPs on the enlarged dataset, and
   (optionally) immediately proposes the next batch.

Run as a script, e.g.:
    python bo_pipeline.py suggest --n 5
    python bo_pipeline.py ingest --file candidates_to_measure.csv --suggest-next 5

See CONFIG below to adjust parameter bounds / which objectives to maximize.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.transforms import unnormalize
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll

torch.set_default_dtype(torch.float64)

# --------------------------------------------------------------------------
# CONFIG -- adjust to your process
# --------------------------------------------------------------------------

PARAM_COLS = [
    "L1", "L2", "L3", "L4",
    "Hex1", "Hex2", "Hex3", "Hex4", "Hex5",
    "RF",
    "Slot2Out2", "Slot2Out3", "Slot2Out4",
]

OBJECTIVE_COLS = [
    "EIC 2,9584 - 3,2623",
    "EIC 18,9014 - 19,4015",
    "EIC 28,9571 - 29,0837",
]

# Lower/upper bound per parameter. Inferred from the observed sample range
# in the uploaded initial dataset (rounded to the instrument's apparent
# [-120, 0] travel range for the 12 stage/RF-output channels, and [0, 400]
# for RF power). ADJUST THESE to your true hardware limits if different.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "L1": (-120.0, 0.0), "L2": (-120.0, 0.0), "L3": (-120.0, 0.0), "L4": (-120.0, 0.0),
    "Hex1": (-120.0, 0.0), "Hex2": (-120.0, 0.0), "Hex3": (-120.0, 0.0),
    "Hex4": (-120.0, 0.0), "Hex5": (-120.0, 0.0),
    "RF": (0.0, 400.0),
    "Slot2Out2": (-120.0, 0.0), "Slot2Out3": (-120.0, 0.0), "Slot2Out4": (-120.0, 0.0),
}

# True -> maximize that objective, False -> minimize. The three EIC channels
# are treated as signals to MAXIMIZE by default (typical for yield/intensity
# optimization). Flip to False for any channel you actually want to suppress
# (e.g. a contaminant/background EIC).
MAXIMIZE_OBJECTIVE = {
    "EIC 2,9584 - 3,2623": True,
    "EIC 18,9014 - 19,4015": True,
    "EIC 28,9571 - 29,0837": True,
}

# EIC intensities span several orders of magnitude (1e-7 to 1e0 in the
# provided data) -> fit the GPs on log10(intensity) rather than raw
# intensity. Strongly recommended for this kind of data; set to False to
# disable per-objective.
LOG_TRANSFORM_OBJECTIVE = {
    "EIC 2,9584 - 3,2623": True,
    "EIC 18,9014 - 19,4015": True,
    "EIC 28,9571 - 29,0837": True,
}

INDEX_COL = "index"          # name given to the leading row-number column
CSV_SEPARATOR = ";"
LOG_EPS = 1e-12               # offset added before log to survive zeros


# --------------------------------------------------------------------------
# Data loading helpers
# --------------------------------------------------------------------------

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the trailing empty ';;;;' columns pandas turns into Unnamed:N,
    and name the leading bare index column."""
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    if df.columns[0] == "Unnamed: 0" or df.columns[0] == "":
        df = df.rename(columns={df.columns[0]: INDEX_COL})
    return df


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the semicolon-separated measurement CSV (German decimal-comma
    is NOT used here -- decimals are '.', columns are ';'-separated, matching
    the uploaded file). Returns a DataFrame with an `index` column plus the
    parameter and objective columns."""
    df = pd.read_csv(csv_path, sep=CSV_SEPARATOR)
    df = _clean_columns(df)
    if INDEX_COL not in df.columns:
        df.insert(0, INDEX_COL, np.arange(1, len(df) + 1))
    return df


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

@dataclass
class MultiObjectiveBOPipeline:
    csv_path: str
    param_cols: Sequence[str] = field(default_factory=lambda: PARAM_COLS)
    objective_cols: Sequence[str] = field(default_factory=lambda: OBJECTIVE_COLS)
    param_bounds: dict = field(default_factory=lambda: PARAM_BOUNDS)
    maximize: dict = field(default_factory=lambda: MAXIMIZE_OBJECTIVE)
    log_transform: dict = field(default_factory=lambda: LOG_TRANSFORM_OBJECTIVE)

    def __post_init__(self):
        self.df = load_dataset(self.csv_path)
        self._validate()
        self.bounds = torch.tensor(
            [self.param_bounds[c] for c in self.param_cols], dtype=torch.float64
        ).T  # shape (2, d)
        self.model: ModelListGP | None = None

    def _validate(self):
        missing = [c for c in list(self.param_cols) + list(self.objective_cols) if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV is missing expected columns: {missing}")

    # ---- tensors -----------------------------------------------------

    def _train_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (X, Y) with X normalized to [0,1]^d and Y transformed so
        that HIGHER IS ALWAYS BETTER (sign-flipped for 'minimize' objectives,
        log10-transformed where configured). Standardization for GP fitting
        is handled internally by each model's Standardize() outcome transform."""
        complete = self.df.dropna(subset=list(self.objective_cols))
        if len(complete) < 2:
            raise ValueError("Need at least 2 fully-measured rows to fit a model.")

        X_raw = torch.tensor(complete[list(self.param_cols)].values, dtype=torch.float64)
        X = (X_raw - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

        Y_cols = []
        for c in self.objective_cols:
            y = complete[c].values.astype(float)
            if self.log_transform.get(c, False):
                y = np.log10(y + LOG_EPS)
            if not self.maximize.get(c, True):
                y = -y
            Y_cols.append(y)
        Y = torch.tensor(np.stack(Y_cols, axis=1), dtype=torch.float64)
        return X, Y

    def _to_internal_objective(self, raw_values: np.ndarray) -> np.ndarray:
        """Apply the same sign-flip / log transform used for training to a
        raw (n, m) array of objective values, e.g. for reporting a Pareto
        front consistently."""
        out = raw_values.copy().astype(float)
        for j, c in enumerate(self.objective_cols):
            if self.log_transform.get(c, False):
                out[:, j] = np.log10(out[:, j] + LOG_EPS)
            if not self.maximize.get(c, True):
                out[:, j] = -out[:, j]
        return out

    # ---- model ---------------------------------------------------------

    def fit(self) -> ModelListGP:
        """Fit one SingleTaskGP with an RBF kernel per objective; combine
        them into a ModelListGP for joint multi-output posterior queries."""
        X, Y = self._train_tensors()
        d = X.shape[-1]
        models = []
        for i in range(Y.shape[-1]):
            covar = ScaleKernel(RBFKernel(ard_num_dims=d))
            gp = SingleTaskGP(
                X,
                Y[:, i : i + 1],
                covar_module=covar,
                input_transform=Normalize(d=d),  # X already in [0,1], harmless/no-op safety net
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            models.append(gp)
        self.model = ModelListGP(*models)
        self._train_Y = Y  # cache for ref point / partitioning
        return self.model

    # ---- acquisition / suggestion --------------------------------------

    def _ref_point(self, Y: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
        """Heuristic reference point: slightly worse than the worst observed
        value per (maximization-oriented) objective."""
        span = Y.max(dim=0).values - Y.min(dim=0).values
        span = torch.where(span > 0, span, torch.ones_like(span))
        return Y.min(dim=0).values - margin * span

    def suggest(
        self,
        n: int = 5,
        mc_samples: int = 256,
        num_restarts: int = 10,
        raw_samples: int = 512,
    ) -> pd.DataFrame:
        """Optimize qEHVI for a batch of `n` candidates. Returns a DataFrame
        in the ORIGINAL (unnormalized) parameter units, ready to run
        experimentally."""
        if self.model is None:
            self.fit()

        ref_point = self._ref_point(self._train_Y)
        partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=self._train_Y)
        # qLogEHVI: numerically-stabilized (log-space) formulation of EHVI,
        # same acquisition semantics as classic qEHVI but recommended by
        # BoTorch to avoid vanishing-gradient issues during optimization.
        acq = qLogExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=ref_point.tolist(),
            partitioning=partitioning,
        )

        unit_bounds = torch.stack(
            [torch.zeros(len(self.param_cols), dtype=torch.float64),
             torch.ones(len(self.param_cols), dtype=torch.float64)]
        )
        candidates_unit, _ = optimize_acqf(
            acq_function=acq,
            bounds=unit_bounds,
            q=n,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=True,
        )
        candidates = unnormalize(candidates_unit, self.bounds)
        df_cand = pd.DataFrame(candidates.detach().numpy(), columns=list(self.param_cols))
        for c in self.objective_cols:
            df_cand[c] = np.nan  # to be filled in after the physical measurement
        return df_cand.round(4)

    # ---- ingesting new measurements -------------------------------------

    def write_candidates(self, df_cand: pd.DataFrame, out_path: str) -> str:
        df_cand.to_csv(out_path, sep=CSV_SEPARATOR, index=False)
        return out_path

    def ingest(self, filled_csv_path: str) -> pd.DataFrame:
        """Read a candidates CSV whose objective columns have been filled in
        with real measurements, append it to the main dataset (with fresh
        index numbers), persist the CSV, and return the updated full
        dataset. Does NOT refit automatically -- call `.fit()` / `.suggest()`
        again, or use `refit_and_suggest`."""
        new_rows = pd.read_csv(filled_csv_path, sep=CSV_SEPARATOR)
        missing = [c for c in list(self.objective_cols) if c not in new_rows.columns]
        if missing:
            raise ValueError(f"Candidates file is missing objective columns: {missing}")
        if new_rows[list(self.objective_cols)].isna().any().any():
            raise ValueError(
                "Some objective values are still empty (NaN) in "
                f"'{filled_csv_path}'. Fill in all measured EIC values before ingesting."
            )

        next_idx = int(self.df[INDEX_COL].max()) + 1
        new_rows = new_rows[list(self.param_cols) + list(self.objective_cols)].copy()
        new_rows.insert(0, INDEX_COL, range(next_idx, next_idx + len(new_rows)))

        self.df = pd.concat([self.df, new_rows], ignore_index=True)
        self.df.to_csv(self.csv_path, sep=CSV_SEPARATOR, index=False)
        return self.df

    def refit_and_suggest(self, n: int = 5) -> pd.DataFrame:
        self.fit()
        return self.suggest(n=n)

    # ---- reporting -------------------------------------------------------

    def pareto_front(self) -> pd.DataFrame:
        """Currently observed Pareto-optimal rows, in original units."""
        complete = self.df.dropna(subset=list(self.objective_cols)).reset_index(drop=True)
        Y_raw = complete[list(self.objective_cols)].values.astype(float)
        Y_int = self._to_internal_objective(Y_raw)  # maximize-oriented

        n = len(Y_int)
        is_efficient = np.ones(n, dtype=bool)
        for i in range(n):
            y = Y_int[i]
            # rows that dominate row i: >= y in every objective, > y in at least one
            dominates_i = np.all(Y_int >= y, axis=1) & np.any(Y_int > y, axis=1)
            dominates_i[i] = False
            if dominates_i.any():
                is_efficient[i] = False  # row i is dominated by something else -> drop it
        return complete.loc[is_efficient].reset_index(drop=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _default_candidates_path(csv_path: str) -> str:
    base, _ = os.path.splitext(csv_path)
    return f"{base}_candidates_to_measure.csv"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=None, help="Path to the main measurement CSV.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("suggest", help="Fit GPs on current data and propose new candidates.")
    s.add_argument("--n", type=int, default=5)
    s.add_argument("--out", default=None, help="Where to write the candidates CSV.")

    i = sub.add_parser("ingest", help="Append filled-in candidate measurements to the main CSV.")
    i.add_argument("--file", required=True, help="Candidates CSV with objective columns filled in.")
    i.add_argument("--suggest-next", type=int, default=0, help="If >0, immediately refit and suggest N more.")

    r = sub.add_parser("pareto", help="Show the current Pareto-optimal measured rows.")

    args = p.parse_args()
    csv_path = args.csv or os.environ.get("BO_CSV_PATH")
    if not csv_path:
        raise SystemExit("Provide --csv path/to/measurements.csv")

    pipe = MultiObjectiveBOPipeline(csv_path=csv_path)

    if args.command == "suggest":
        df_cand = pipe.suggest(n=args.n)
        out = args.out or _default_candidates_path(csv_path)
        pipe.write_candidates(df_cand, out)
        print(f"Wrote {len(df_cand)} candidates to {out}")
        print(df_cand.to_string(index=False))

    elif args.command == "ingest":
        updated = pipe.ingest(args.file)
        print(f"Ingested measurements. Dataset now has {len(updated)} rows -> saved to {csv_path}")
        if args.suggest_next > 0:
            df_cand = pipe.refit_and_suggest(n=args.suggest_next)
            out = _default_candidates_path(csv_path)
            pipe.write_candidates(df_cand, out)
            print(f"Refit complete. Wrote {len(df_cand)} new candidates to {out}")
            print(df_cand.to_string(index=False))

    elif args.command == "pareto":
        front = pipe.pareto_front()
        # Objective values here can genuinely span ~1e-7 to ~1 (raw EIC
        # intensities). Plain to_string() rounds to 6 decimals and makes
        # small-but-real values print as 0.000000, so use general/scientific
        # formatting instead -- this changes only how numbers are displayed,
        # not any value used internally by the model.
        with pd.option_context("display.float_format", lambda x: f"{x:.4g}"):
            print(front.to_string(index=False))


if __name__ == "__main__":
    main()
