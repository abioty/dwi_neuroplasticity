"""Pipeline helpers for repair-vs-compensation analyses in VPT diffusion MRI cohorts.

Concept translation used throughout the code:
- Repair = normalization toward FT-like white matter structure.
- Compensation = non-FT-like enhancement or reorganization in alternative pathways.
- Kidokoro moderation = injury severity changes the balance between repair and compensation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


FBA_METRICS: tuple[str, str, str] = ("FD", "FC", "FDC")
VPT_GROUPS: tuple[str, str, str] = ("SP", "TP", "BTP")
ALL_GROUPS: tuple[str, str, str, str] = ("FT", "SP", "TP", "BTP")


@dataclass(frozen=True)
class GroupThresholds:
    """Thresholds for VPT cognitive performance groups.

    NOTE: Verify these boundaries against Introduction/Methods before locking analysis.
    """

    sp_lower_exclusive: float = 107.0
    tp_lower_inclusive: float = 93.0
    tp_upper_inclusive: float = 107.0


@dataclass(frozen=True)
class AnalysisConfig:
    """Runtime options for repair/compensation computations."""

    thresholds: GroupThresholds = GroupThresholds()
    ft_group_label: str = "FT"
    vpt_label: str = "VPT"
    id_col: str = "subject_id"
    gca_col: str = "gca"
    term_col: str = "term_status"
    group_col: str = "group"
    kidokoro_col: str = "kidokoro"
    tract_col: str = "tract"


def assign_performance_group(
    metadata: pd.DataFrame,
    config: AnalysisConfig = AnalysisConfig(),
) -> pd.DataFrame:
    """Assign SP/TP/BTP/FT labels from GCA and term status.

    FT controls remain a separate comparison group and are never recoded as VPT.
    """

    df = metadata.copy()
    required = {config.id_col, config.term_col, config.gca_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

    def _label_row(row: pd.Series) -> str:
        if row[config.term_col] == config.ft_group_label:
            return config.ft_group_label

        gca = row[config.gca_col]
        if pd.isna(gca):
            return "UNKNOWN"
        if gca > config.thresholds.sp_lower_exclusive:
            return "SP"
        if config.thresholds.tp_lower_inclusive <= gca <= config.thresholds.tp_upper_inclusive:
            return "TP"
        return "BTP"

    df[config.group_col] = df.apply(_label_row, axis=1)
    return df


def validate_expected_counts(
    metadata: pd.DataFrame,
    expected_counts: dict[str, int] | None = None,
    config: AnalysisConfig = AnalysisConfig(),
) -> dict[str, int]:
    """Validate cohort counts against the protocol and return observed counts."""

    if expected_counts is None:
        expected_counts = {"SP": 42, "TP": 73, "BTP": 66, "FT": 45}

    observed = metadata[config.group_col].value_counts(dropna=False).to_dict()
    mismatches = {
        g: (observed.get(g, 0), n)
        for g, n in expected_counts.items()
        if observed.get(g, 0) != n
    }
    if mismatches:
        details = ", ".join([f"{g}: observed={o}, expected={e}" for g, (o, e) in mismatches.items()])
        raise ValueError(f"Cohort count mismatch -> {details}")
    return {g: observed.get(g, 0) for g in expected_counts}


def build_ft_reference(
    tract_metrics: pd.DataFrame,
    metric_cols: Iterable[str] = FBA_METRICS,
    config: AnalysisConfig = AnalysisConfig(),
) -> pd.DataFrame:
    """Build FT tract-level reference distributions (mean, SD, percentile anchors)."""

    needed = {config.id_col, config.group_col, config.tract_col, *metric_cols}
    missing = needed - set(tract_metrics.columns)
    if missing:
        raise ValueError(f"Missing required tract metric columns: {sorted(missing)}")

    ft = tract_metrics.loc[tract_metrics[config.group_col] == config.ft_group_label].copy()
    if ft.empty:
        raise ValueError("No FT rows found; cannot build FT reference distribution.")

    refs: list[pd.DataFrame] = []
    for metric in metric_cols:
        summary = (
            ft.groupby(config.tract_col)[metric]
            .agg(
                ft_mean="mean",
                ft_sd="std",
                ft_p10=lambda s: np.nanpercentile(s, 10),
                ft_p50=lambda s: np.nanpercentile(s, 50),
                ft_p90=lambda s: np.nanpercentile(s, 90),
            )
            .reset_index()
        )
        summary["metric"] = metric
        refs.append(summary)

    return pd.concat(refs, ignore_index=True)


def add_ft_referenced_zscores(
    tract_metrics: pd.DataFrame,
    ft_reference: pd.DataFrame,
    config: AnalysisConfig = AnalysisConfig(),
) -> pd.DataFrame:
    """Attach FT-referenced z-scores for each participant x tract x metric."""

    long = tract_metrics.melt(
        id_vars=[config.id_col, config.group_col, config.kidokoro_col, config.tract_col],
        value_vars=list(FBA_METRICS),
        var_name="metric",
        value_name="value",
    )
    merged = long.merge(ft_reference, on=[config.tract_col, "metric"], how="left")

    merged["ft_sd"] = merged["ft_sd"].replace({0.0: np.nan})
    merged["z_ft"] = (merged["value"] - merged["ft_mean"]) / merged["ft_sd"]
    merged["abs_z_ft"] = merged["z_ft"].abs()
    return merged


def compute_repair_index(
    zscores: pd.DataFrame,
    injury_vulnerable_tracts: Iterable[str],
    config: AnalysisConfig = AnalysisConfig(),
) -> pd.DataFrame:
    """Compute repair index as closeness to FT in injury-vulnerable tracts.

    Higher index = more FT-like (better repair). Based on negative mean absolute z.
    """

    tract_set = set(injury_vulnerable_tracts)
    subset = zscores[zscores[config.tract_col].isin(tract_set)].copy()
    if subset.empty:
        raise ValueError("No matching injury-vulnerable tracts found in z-score table.")

    out = (
        subset.groupby([config.id_col, config.group_col], as_index=False)["abs_z_ft"]
        .mean()
        .rename(columns={"abs_z_ft": "mean_abs_z_vulnerable"})
    )
    out["repair_index"] = -out["mean_abs_z_vulnerable"]
    return out


def compute_compensation_index(
    zscores: pd.DataFrame,
    alternative_tracts: Iterable[str],
    supra_typical_threshold_z: float = 1.0,
    config: AnalysisConfig = AnalysisConfig(),
) -> pd.DataFrame:
    """Compute compensation index from supra-typical elevation in alternative tracts.

    Operationalization:
    - keep only candidate alternative pathways,
    - zero out negative z-values (below FT mean),
    - summarize the positive tail with average excess above threshold.
    """

    tract_set = set(alternative_tracts)
    subset = zscores[zscores[config.tract_col].isin(tract_set)].copy()
    if subset.empty:
        raise ValueError("No matching alternative tracts found in z-score table.")

    subset["supra_typical_excess"] = (subset["z_ft"] - supra_typical_threshold_z).clip(lower=0)
    out = (
        subset.groupby([config.id_col, config.group_col], as_index=False)["supra_typical_excess"]
        .mean()
        .rename(columns={"supra_typical_excess": "compensation_index"})
    )
    return out


def fit_group_and_moderation_models(
    zscores: pd.DataFrame,
    config: AnalysisConfig = AnalysisConfig(),
) -> dict[str, object]:
    """Fit core models including group effects and Group x Kidokoro interaction.

    Model:
      z_ft ~ C(group) + kidokoro + C(group):kidokoro + C(tract) + C(metric)
    """

    model_df = zscores.dropna(subset=["z_ft", config.kidokoro_col, config.group_col]).copy()
    if model_df.empty:
        raise ValueError("No complete rows available for modeling.")

    formula = (
        f"z_ft ~ C({config.group_col}) + {config.kidokoro_col} + "
        f"C({config.group_col}):{config.kidokoro_col} + C({config.tract_col}) + C(metric)"
    )

    fit = smf.ols(formula=formula, data=model_df).fit()
    anova = smf.ols(formula=formula, data=model_df).fit().summary2().tables[1]
    return {"fit": fit, "coefficients": anova}


def build_continuum_table(
    metadata_with_group: pd.DataFrame,
    repair_index: pd.DataFrame,
    compensation_index: pd.DataFrame,
    config: AnalysisConfig = AnalysisConfig(),
) -> pd.DataFrame:
    """Create participant-level repair-compensation continuum outputs."""

    continuum = (
        metadata_with_group[[config.id_col, config.group_col, config.kidokoro_col]]
        .drop_duplicates()
        .merge(repair_index[[config.id_col, "repair_index"]], on=config.id_col, how="left")
        .merge(compensation_index[[config.id_col, "compensation_index"]], on=config.id_col, how="left")
    )
    continuum["repair_minus_compensation"] = (
        continuum["repair_index"] - continuum["compensation_index"]
    )
    return continuum


def run_end_to_end(
    metadata: pd.DataFrame,
    tract_metrics: pd.DataFrame,
    injury_vulnerable_tracts: Iterable[str],
    alternative_tracts: Iterable[str],
    config: AnalysisConfig = AnalysisConfig(),
) -> dict[str, pd.DataFrame | dict[str, object]]:
    """Execute full workflow and return analysis-ready artifacts."""

    meta = assign_performance_group(metadata, config=config)
    validate_expected_counts(meta, config=config)

    merged_metrics = tract_metrics.merge(
        meta[[config.id_col, config.group_col, config.kidokoro_col]],
        on=config.id_col,
        how="inner",
    )

    ft_ref = build_ft_reference(merged_metrics, config=config)
    zscores = add_ft_referenced_zscores(merged_metrics, ft_ref, config=config)
    repair = compute_repair_index(zscores, injury_vulnerable_tracts, config=config)
    compensation = compute_compensation_index(zscores, alternative_tracts, config=config)
    models = fit_group_and_moderation_models(zscores, config=config)
    continuum = build_continuum_table(meta, repair, compensation, config=config)

    return {
        "metadata": meta,
        "ft_reference": ft_ref,
        "zscores": zscores,
        "repair_index": repair,
        "compensation_index": compensation,
        "continuum": continuum,
        "models": models,
    }
