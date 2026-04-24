"""Train a strength model from historical World Cup data.

We regress the `round_score` (1=group stage, 6=winner) of each team at each
World Cup on its features. The resulting prediction is interpreted as an
expected "tournament strength" which we then feed into a match-level
simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from .features import FEATURE_COLUMNS, load_historical, load_2026


@dataclass
class TrainedModel:
    ridge: Pipeline
    gbm: GradientBoostingRegressor
    cv_mae: float
    feature_importance: pd.Series


def _make_ridge() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=3.0, random_state=42)),
    ])


def _make_gbm() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )


def train_strength_model(historical: pd.DataFrame) -> TrainedModel:
    X = historical[FEATURE_COLUMNS].to_numpy()
    y = historical["round_score"].to_numpy()
    groups = historical["year"].to_numpy()

    unique_years = np.unique(groups)
    n_splits = min(5, len(unique_years))
    gkf = GroupKFold(n_splits=n_splits)
    fold_errors = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        ridge = _make_ridge().fit(X[train_idx], y[train_idx])
        gbm = _make_gbm().fit(X[train_idx], y[train_idx])
        preds = 0.6 * ridge.predict(X[test_idx]) + 0.4 * gbm.predict(X[test_idx])
        fold_errors.append(mean_absolute_error(y[test_idx], preds))

    cv_mae = float(np.mean(fold_errors))
    ridge = _make_ridge().fit(X, y)
    gbm = _make_gbm().fit(X, y)

    # Coefficient magnitudes from Ridge as interpretable importance proxy.
    ridge_coefs = pd.Series(ridge.named_steps["ridge"].coef_, index=FEATURE_COLUMNS).abs()
    ridge_imp = ridge_coefs / ridge_coefs.sum()
    gbm_imp = pd.Series(gbm.feature_importances_, index=FEATURE_COLUMNS)
    blended = (0.6 * ridge_imp + 0.4 * gbm_imp).sort_values(ascending=False)

    return TrainedModel(ridge=ridge, gbm=gbm, cv_mae=cv_mae, feature_importance=blended)


def score_2026(trained: TrainedModel, df_2026: pd.DataFrame) -> pd.DataFrame:
    X = df_2026[FEATURE_COLUMNS].to_numpy()
    ridge_pred = trained.ridge.predict(X)
    gbm_pred = trained.gbm.predict(X)
    ml_strength = 0.6 * ridge_pred + 0.4 * gbm_pred

    # Bayesian-style blend with a FIFA-rank prior: rank 1 -> 5.5, rank 48 -> 2.0.
    rank = df_2026["fifa_rank"].to_numpy()
    rank_prior = 5.5 - 3.5 * (np.log1p(rank) / np.log1p(48))

    strength = 0.55 * ml_strength + 0.45 * rank_prior

    out = df_2026[["team", "confederation", "fifa_rank"]].copy()
    out["ml_strength"] = ml_strength
    out["rank_prior"] = rank_prior
    out["strength"] = strength
    out["win_rate_proxy"] = (strength - strength.min()) / (strength.max() - strength.min())
    return out.sort_values("strength", ascending=False).reset_index(drop=True)


def load_all(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return load_historical(data_dir), load_2026(data_dir)
