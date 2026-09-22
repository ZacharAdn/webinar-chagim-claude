"""Rung 2 -- the prediction, its baseline, and the leakage demo.

Everything here is deterministic (fixed seed) so the numbers on stage are the
numbers from the rehearsal.

Three functions matter:
  train_model()      -- the honest pipeline: split first, then fit.
  baseline via DummyClassifier -- what "predict nobody" scores. The number the
                        model has to beat before anyone should care about it.
  leakage_demo()     -- the same model class on two different splits. Oversample
                        before the split and duplicate rows land on both sides,
                        so the test score measures memory, not prediction.

This module takes a Spec and never imports the ladder runner, so the app runs
from a fresh clone with nothing but src/ on the path.
"""

from __future__ import annotations

import hashlib
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
from pandas.api import types as ptypes
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from insights import split_columns as column_split  # noqa: E402
from insights import to_snake  # noqa: E402

MODEL_VERSION = "logreg-v1"
ESTIMATORS = {"logreg": "Logistic regression", "tree": "Decision tree"}
MAX_CATEGORIES = 50       # above this an object column is an identifier


@dataclass(frozen=True)
class Spec:
    """Everything the model needs from ladder.toml, and nothing more."""

    target: str
    id_column: str
    positive_label: str
    seed: int = 42
    test_size: float = 0.2
    categorical_int_max_unique: int = 10
    estimator: str = "logreg"

    @classmethod
    def from_config(cls, cfg) -> "Spec":
        """From a LadderConfig -- how the runner builds it."""
        return cls(
            target=to_snake(cfg.target_column),
            id_column=to_snake(cfg.id_column) if cfg.id_column else "row_id",
            positive_label=cfg.positive_label,
            seed=cfg.seed,
            test_size=cfg.test_size,
            categorical_int_max_unique=cfg.categorical_int_max_unique,
            estimator=getattr(cfg, "estimator", "logreg"),
        )

    @classmethod
    def from_toml(cls, root: Path | None = None) -> "Spec":
        """Straight from ladder.toml -- how the app builds it, with no runner."""
        with (Path(root or Path(__file__).resolve().parents[1]) / "ladder.toml").open(
            "rb"
        ) as fh:
            raw = tomllib.load(fh)
        dataset = raw["dataset"]
        model = raw.get("model", {})
        return cls(
            target=to_snake(dataset["target_column"]),
            id_column=to_snake(dataset.get("id_column", "") or "") or "row_id",
            positive_label=str(dataset["positive_label"]),
            seed=int(model.get("seed", 42)),
            test_size=float(model.get("test_size", 0.2)),
            categorical_int_max_unique=int(
                model.get("categorical_int_max_unique", 10)
            ),
            estimator=str(model.get("estimator", "logreg")),
        )


@dataclass
class Metrics:
    """One row of the scoreboard."""

    name: str
    accuracy: float
    precision: float
    recall: float
    roc_auc: float
    n_test: int = 0
    note: str = ""

    def as_row(self) -> dict:
        return {
            "model": self.name,
            "accuracy": round(self.accuracy, 3),
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "roc_auc": round(self.roc_auc, 3),
            "note": self.note,
        }


@dataclass
class TrainedModel:
    pipeline: Pipeline
    metrics: Metrics
    baseline: Metrics
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    version: str = MODEL_VERSION
    origin: str = "trained in this process"


def split_columns(df: pd.DataFrame, spec: Spec) -> tuple[list[str], list[str]]:
    """Numeric versus categorical -- insights.py owns the rule, this adds one guard.

    A float is a measurement. An integer with only a handful of distinct values
    is a flag and belongs in the encoder. On top of that shared rule, a
    categorical column with more distinct values than MAX_CATEGORIES is a
    second identifier and is dropped here, not one-hot encoded into thousands
    of columns -- a modelling concern, which is why it lives in this file and
    not in insights.py.
    """
    numeric, categorical = column_split(
        df, spec.id_column, spec.target, spec.categorical_int_max_unique
    )
    categorical = [
        c for c in categorical if df[c].nunique(dropna=True) <= MAX_CATEGORIES
    ]
    return numeric, categorical


def _xy(df: pd.DataFrame, spec: Spec) -> tuple[pd.DataFrame, pd.Series, list, list]:
    """Sorted by the id column first, so every caller gets the same split.

    train_test_split shuffles positions from a fixed seed, which quietly makes
    the split a function of the incoming row order. The CSV arrives in file
    order and Supabase in whatever order PostgREST chose, so without this line
    the app quoted recall 0.540 where REPORT.md quoted 0.559 -- same rows, same
    model, same seed. A demo cannot survive that.
    """
    if spec.id_column in df.columns:
        df = df.sort_values(spec.id_column, kind="stable").reset_index(drop=True)
    y = (df[spec.target].astype(str).str.strip() == spec.positive_label).astype(int)
    numeric, categorical = split_columns(df, spec)
    return df[numeric + categorical], y, numeric, categorical


def _build_pipeline(spec: Spec, numeric: list[str], categorical: list[str],
                    estimator=None) -> Pipeline:
    """The column lists are passed in, never re-derived from the frame.

    Deriving them here from dtypes would call senior_citizen numeric -- it is an
    integer column -- and scale a 0/1 flag instead of encoding it. That moved the
    honest leakage recall from 0.495 to 0.505 against the reference run.
    """
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical,
            ),
        ]
    )
    if estimator is None:
        estimator = _estimator_from(spec)
    return Pipeline([("pre", pre), ("clf", estimator)])


def _estimator_from(spec: Spec):
    """The family the rung-2 gate chose, straight out of ladder.toml.

    A tree is capped at depth 5 on purpose: an uncapped tree memorises the
    training half and the number stops meaning anything, which is the lesson of
    the next rung, not of this one.
    """
    if spec.estimator == "tree":
        return DecisionTreeClassifier(max_depth=5, random_state=spec.seed)
    return LogisticRegression(max_iter=1000, random_state=spec.seed)


def _score(name: str, model, x_test, y_test, note: str = "") -> Metrics:
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]
    return Metrics(
        name=name,
        accuracy=accuracy_score(y_test, pred),
        precision=precision_score(y_test, pred, zero_division=0),
        recall=recall_score(y_test, pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, proba),
        n_test=len(y_test),
        note=note,
    )


def train_model(df: pd.DataFrame, spec: Spec) -> TrainedModel:
    """Split first, fit second. The boring order is the correct one."""
    x, y, numeric, categorical = _xy(df, spec)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=spec.test_size, random_state=spec.seed, stratify=y
    )
    pipeline = _build_pipeline(spec, numeric, categorical)
    pipeline.fit(x_train, y_train)
    metrics = _score(
        ESTIMATORS.get(spec.estimator, spec.estimator),
        pipeline, x_test, y_test, "split before fit",
    )

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    baseline = _score(
        "Baseline: nobody is positive",
        dummy,
        x_test,
        y_test,
        "predicts the majority class",
    )

    return TrainedModel(
        pipeline=pipeline,
        metrics=metrics,
        baseline=baseline,
        numeric=numeric,
        categorical=categorical,
        version=f"{spec.estimator}-{_fingerprint(pipeline, x_test)}",
    )


def _fingerprint(pipeline: Pipeline, x_test: pd.DataFrame) -> str:
    """Eight hex characters that change exactly when the model's answers do.

    A constant "v1" told the predictions table nothing: retrain on a changed
    table and every row still said v1. Hashing the test-set probabilities means
    the same model always carries the same version, wherever it was trained.
    """
    probabilities = pipeline.predict_proba(x_test)[:, 1].round(6)
    return hashlib.sha1(probabilities.tobytes()).hexdigest()[:8]


MODEL_DIR = "models"


def model_path(root: Path, estimator: str) -> Path:
    return Path(root) / MODEL_DIR / f"{estimator}.joblib"


def save_model(trained: TrainedModel, path: Path) -> Path:
    """The whole TrainedModel to one small file, 8KB for the logistic regression."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained, path)
    return path


def load_model(path: Path) -> TrainedModel | None:
    """The saved model, or None when there is none or it will not load.

    A pickle is tied to the scikit-learn that wrote it. Rather than crash the
    app on a library upgrade, an unreadable file counts as a missing one and
    the caller trains again.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        loaded = joblib.load(path)
    except Exception:  # noqa: BLE001 -- any unpickling failure means "retrain"
        return None
    return loaded if isinstance(loaded, TrainedModel) else None


def feature_defaults(df: pd.DataFrame, spec: Spec) -> dict:
    """A complete record built out of the table itself.

    The scoring form on rung 2 asks for a handful of columns, not nineteen. The
    rest have to come from somewhere, and the table is the only honest source:
    the median for a measurement, the most common value for a category. The app
    shows what was filled in, so the defaulting is on screen rather than hidden.
    """
    numeric, categorical = split_columns(df, spec)
    defaults: dict = {}
    for column in numeric:
        defaults[column] = float(df[column].median())
    for column in categorical:
        modes = df[column].mode(dropna=True)
        defaults[column] = modes.iloc[0] if not modes.empty else ""
    return defaults


def score_record(trained: TrainedModel, record: dict) -> float:
    """P(positive) for one record, from a model that is already trained.

    The pipeline carries the imputer, the scaler and the encoder, so a raw dict
    of human-typed values is all this needs -- and a column the caller left out
    arrives as NaN, which the imputer handles the same way it did in training.
    """
    columns = trained.numeric + trained.categorical
    row = pd.DataFrame([{column: record.get(column) for column in columns}])
    return float(trained.pipeline.predict_proba(row)[0, 1])


def leakage_demo(df: pd.DataFrame, spec: Spec) -> tuple[Metrics, Metrics]:
    """The same decision tree, two splits, two numbers that are far apart.

    Honest: split, then oversample the training half only.
    Leaky : oversample the whole table, then split -- copies of the same row end
            up in train and in test, so the tree recognises rows it has already
            memorised.
    """
    x, y, numeric, categorical = _xy(df, spec)
    label = "__target__"

    # Honest run.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=spec.test_size, random_state=spec.seed, stratify=y
    )
    train = x_train.copy()
    train[label] = y_train.values
    minority = train[train[label] == 1]
    balanced = pd.concat(
        [
            train,
            minority.sample(
                len(train[train[label] == 0]) - len(minority),
                replace=True,
                random_state=spec.seed,
            ),
        ]
    )
    honest_model = _build_pipeline(
        spec, numeric, categorical, DecisionTreeClassifier(random_state=spec.seed)
    )
    honest_model.fit(balanced.drop(columns=[label]), balanced[label])
    honest = _score(
        "Decision tree, split before oversampling",
        honest_model,
        x_test,
        y_test,
        "the number you can take to a meeting",
    )

    # Leaky run.
    full = x.copy()
    full[label] = y.values
    minority_full = full[full[label] == 1]
    oversampled = pd.concat(
        [
            full,
            minority_full.sample(
                len(full[full[label] == 0]) - len(minority_full),
                replace=True,
                random_state=spec.seed,
            ),
        ]
    )
    lx = oversampled.drop(columns=[label])
    ly = oversampled[label]
    lx_train, lx_test, ly_train, ly_test = train_test_split(
        lx, ly, test_size=spec.test_size, random_state=spec.seed, stratify=ly
    )
    leaky_model = _build_pipeline(
        spec, numeric, categorical, DecisionTreeClassifier(random_state=spec.seed)
    )
    leaky_model.fit(lx_train, ly_train)
    leaky = _score(
        "Decision tree, oversampling before split",
        leaky_model,
        lx_test,
        ly_test,
        "duplicate rows on both sides -- this measures memory",
    )
    return honest, leaky


def score_records(model: TrainedModel, df: pd.DataFrame, spec: Spec,
                  extra_columns: list[str] | None = None) -> pd.DataFrame:
    """Probability per row, highest first, with a few columns for context."""
    x, _, _, _ = _xy(df, spec)
    proba = model.pipeline.predict_proba(x)[:, 1]
    ids = (
        df[spec.id_column]
        if spec.id_column in df.columns
        else pd.Series(range(1, len(df) + 1))
    )
    out = pd.DataFrame({spec.id_column: ids.values, "probability": proba})
    for column in extra_columns or []:
        if column in df.columns:
            out[column] = df[column].values
    return out.sort_values("probability", ascending=False).reset_index(drop=True)
