import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_config_study_bundle(
    longshort_df: pd.DataFrame,
    long_df: pd.DataFrame,
    null_df: pd.DataFrame,
    out_dir: str,
    top_k: int = 25,
    objective: str = "eae",
):
    """Orchestrates the config study.

    Args:
        longshort_df: Long/Short strategy results DataFrame
        long_df: Long-only strategy results DataFrame
        null_df: Null run results DataFrame
        out_dir: Output directory
        top_k: Number of top configurations to analyze
        objective: Objective metric to optimize

    Returns:
        None
    """

    longshort = "longshort"
    long = "long"
    null = "null"

    # Create output directories
    longshort_dir = os.path.join(out_dir, longshort)
    long_dir = os.path.join(out_dir, long)
    null_dir = os.path.join(out_dir, null)
    os.makedirs(longshort_dir, exist_ok=True)
    os.makedirs(long_dir, exist_ok=True)
    os.makedirs(null_dir, exist_ok=True)

    # Aggregate null runs by config
    config_cols = [
        "bull_threshold",
        "bear_threshold",
        "return_mode",
        "wait_weeks",
        "hold_weeks",
        "strategy",
    ]
    null_df = null_df.groupby(config_cols, as_index=False).mean(numeric_only=True)

    run_general_study(longshort_df, longshort_dir, longshort)
    run_general_study(long_df, long_dir, long)
    run_general_study(null_df, null_dir, null)

    run_bull_bear_analysis(longshort_df, longshort_dir, longshort)
    run_bull_bear_analysis(long_df, long_dir, long)
    run_bull_bear_analysis(null_df, null_dir, null)

    run_hold_wait_analysis(longshort_df, longshort_dir, longshort)
    run_hold_wait_analysis(long_df, long_dir, long)
    run_hold_wait_analysis(null_df, null_dir, null)

    run_return_mode_analysis(longshort_df, longshort_dir, longshort)
    run_return_mode_analysis(long_df, long_dir, long)
    run_return_mode_analysis(null_df, null_dir, null)

    run_eae_analysis(longshort_df, longshort_dir, longshort, top_k=top_k)
    run_eae_analysis(long_df, long_dir, long, top_k=top_k)
    run_eae_analysis(null_df, null_dir, null, top_k=top_k)

    analyze_timing_structure_stability(
        longshort_df, longshort_dir, longshort, objective="eae"
    )
    analyze_timing_structure_stability(long_df, long_dir, long, objective="eae")
    analyze_timing_structure_stability(null_df, null_dir, null, objective="eae")

    analyze_local_parameter_robustness(
        longshort_df, longshort_dir, longshort, objective="eae", top_k=3
    )
    analyze_local_parameter_robustness(
        long_df, long_dir, long, objective="eae", top_k=3
    )
    analyze_local_parameter_robustness(
        null_df, null_dir, null, objective="eae", top_k=3
    )

    analyze_trade_count_dependence(
        longshort_df, longshort_dir, longshort, objective="eae"
    )
    analyze_trade_count_dependence(long_df, long_dir, long, objective="eae")
    analyze_trade_count_dependence(null_df, null_dir, null, objective="eae")

    analyze_signal_quality_relationship(
        longshort_df, longshort_dir, longshort, objective="eae"
    )
    analyze_signal_quality_relationship(long_df, long_dir, long, objective="eae")
    analyze_signal_quality_relationship(null_df, null_dir, null, objective="eae")

    compare_long_short_vs_long_only(
        longshort_df, long_df, longshort_dir, objective="eae"
    )

    analyze_null_dominance(
        longshort_df, null_df, longshort_dir, longshort, objective="eae", top_k=3
    )

    analyze_multi_objective_tradeoffs(
        longshort_df, longshort_dir, longshort, objective="eae"
    )
    analyze_multi_objective_tradeoffs(long_df, long_dir, long, objective="eae")
    analyze_multi_objective_tradeoffs(null_df, null_dir, null, objective="eae")

    build_deployment_shortlist(
        longshort_df, longshort_dir, longshort, objective="eae", top_n=3
    )
    build_deployment_shortlist(long_df, long_dir, long, objective="eae", top_n=3)
    build_deployment_shortlist(null_df, null_dir, null, objective="eae", top_n=3)

    pareto_front(
        longshort_df,
        longshort_dir,
        longshort,
        maximize_cols=["eae"],
        minimize_cols=["max_drawdown"],
    )
    pareto_front(
        long_df, long_dir, long, maximize_cols=["eae"], minimize_cols=["max_drawdown"]
    )


def run_general_study(df: pd.DataFrame, dir: str, strategy_name: str):
    """Run general study analysis including histograms and visualizations.

    Args:
        df: DataFrame with strategy results
        dir: Directory for results
        strategy_name: Name of the strategy

    Returns:
        None
    """

    dir = os.path.join(dir, "general_study")
    os.makedirs(dir, exist_ok=True)

    # Define histogram columns and make histograms
    histogram_columns = [
        "eae",
        "max_drawdown",
        "profit_factor",
        "win_rate",
        "cum_return_pct",
        "trades",
        "total_signal_numeric_mean",
        "long_signal_ic",
        "short_signal_ic",
        "trades_ic",
    ]

    lines = []
    for col in histogram_columns:
        if col not in df.columns:
            continue
        s = finite_series(df, col)
        if s.empty:
            continue

        plt.figure(figsize=(10, 6))
        plt.hist(s, bins=50)
        plt.axvline(
            s.mean(), color="red", linestyle="--", label=f"Mean: {s.mean():.2f}"
        )
        plt.axvline(
            s.median(), color="green", linestyle="--", label=f"Median: {s.median():.2f}"
        )
        plt.axvline(
            s.quantile(0.25),
            color="blue",
            linestyle="--",
            label=f"25th Per: {s.quantile(0.25):.2f}",
        )
        plt.axvline(
            s.quantile(0.75),
            color="orange",
            linestyle="--",
            label=f"75th Per: {s.quantile(0.75):.2f}",
        )
        plt.legend()
        plt.title(f"{col} for {strategy_name}")
        plt.savefig(os.path.join(dir, f"{strategy_name}_{col}.png"))
        plt.close()
        lines.append(
            f"{col}: mean={s.mean():.2f}, "
            f"median={s.median():.2f}, "
            f"25th={s.quantile(0.25):.2f}, "
            f"75th={s.quantile(0.75):.2f}"
        )

    with open(os.path.join(dir, f"{strategy_name}_stats.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def run_bull_bear_analysis(df: pd.DataFrame, dir: str, strategy_name: str):
    """Run bull/bear threshold analysis including heatmaps and best-threshold summaries.

    Args:
        df: DataFrame with strategy results
        dir: Directory for results
        strategy_name: Name of the strategy

    Returns:
        None
    """

    dir = os.path.join(dir, "bull_bear_analysis")
    os.makedirs(dir, exist_ok=True)

    required_cols = ["bull_threshold", "bear_threshold", "eae"]
    for c in required_cols:
        if c not in df.columns:
            return

    d = df.copy()
    d["bull_threshold"] = as_numeric(d["bull_threshold"])
    d["bear_threshold"] = as_numeric(d["bear_threshold"])
    d["eae"] = as_numeric(d["eae"])
    d = finite_frame(d, ["bull_threshold", "bear_threshold", "eae"])
    if d.empty:
        return

    # Heatmap: mean EAE over bull x bear
    pivot = d.pivot_table(
        index="bull_threshold", columns="bear_threshold", values="eae", aggfunc="mean"
    )
    if not pivot.empty:
        plt.figure(figsize=(10, 6))
        plt.imshow(pivot.values, aspect="auto")
        plt.colorbar(label="eae")
        plt.title(f"eae by Bull/Bear for {strategy_name}")
        plt.xlabel("Bear Threshold")
        plt.ylabel("Bull Threshold")
        plt.xticks(range(len(pivot.columns)), [str(int(c)) for c in pivot.columns])
        plt.yticks(range(len(pivot.index)), [str(int(i)) for i in pivot.index])
        plt.tight_layout()
        plt.savefig(os.path.join(dir, f"{strategy_name}_heat_bull_bear_eae.png"))
        plt.close()

    # Best bull/bear pair (by mean EAE across all other dimensions)
    bb = (
        d.groupby(["bull_threshold", "bear_threshold"], as_index=False)
        .agg(
            eae_mean=("eae", "mean"),
            eae_median=("eae", "median"),
            trades_median=(
                ("trades", "median") if "trades" in d.columns else ("eae", "size")
            ),
        )
        .sort_values("eae_mean", ascending=False)
    )
    if not bb.empty:
        best = bb.iloc[0]
        out_txt = os.path.join(dir, f"{strategy_name}_bull_bear_summary.txt")
        with open(out_txt, "w") as f:
            f.write("best_bull_bear_by_mean_eae:\n")
            f.write(
                f"bull={int(best['bull_threshold'])}, "
                f"bear={int(best['bear_threshold'])}, "
                f"eae_mean={best['eae_mean']:.6f}, "
                f"eae_median={best['eae_median']:.6f}, "
                f"trades_median={best['trades_median']:.3f}\n\n"
            )
            f.write("top_25_bull_bear_pairs_by_mean_eae:\n")
            f.write(bb.head(25).to_string(index=False) + "\n")


def run_hold_wait_analysis(df: pd.DataFrame, dir: str, strategy_name: str):
    """Run hold/wait timing analysis including heatmaps and best-timing summaries.

    Args:
        df: DataFrame with strategy results
        dir: Directory for results
        strategy_name: Name of the strategy

    Returns:
        None
    """

    dir = os.path.join(dir, "hold_wait_analysis")
    os.makedirs(dir, exist_ok=True)

    required_cols = ["hold_weeks", "wait_weeks", "eae"]
    for c in required_cols:
        if c not in df.columns:
            return

    d = df.copy()
    d["hold_weeks"] = as_numeric(d["hold_weeks"])
    d["wait_weeks"] = as_numeric(d["wait_weeks"])
    d["eae"] = as_numeric(d["eae"])
    d = finite_frame(d, ["hold_weeks", "wait_weeks", "eae"])
    if d.empty:
        return

    # Heatmap: median EAE over hold x wait
    hw_med = d.groupby(["hold_weeks", "wait_weeks"], as_index=False).agg(
        eae_median=("eae", "median"),
        eae_mean=("eae", "mean"),
        trades_median=(
            ("trades", "median") if "trades" in d.columns else ("eae", "size")
        ),
    )
    pivot_med = hw_med.pivot_table(
        index="hold_weeks", columns="wait_weeks", values="eae_median", aggfunc="mean"
    )
    if not pivot_med.empty:
        plt.figure(figsize=(10, 6))
        plt.imshow(pivot_med.values, aspect="auto")
        plt.colorbar(label="median eae")
        plt.title(f"Median eae by Hold/Wait for {strategy_name}")
        plt.xlabel("Wait Weeks")
        plt.ylabel("Hold Weeks")
        plt.xticks(
            range(len(pivot_med.columns)), [str(int(c)) for c in pivot_med.columns]
        )
        plt.yticks(range(len(pivot_med.index)), [str(int(i)) for i in pivot_med.index])
        plt.tight_layout()
        plt.savefig(os.path.join(dir, f"{strategy_name}_heat_hold_wait_median_eae.png"))
        plt.close()

    # Heatmap: median trades over hold x wait (helps diagnose sparse-trade pockets)
    if "trades" in d.columns:
        hw_tr = d.groupby(["hold_weeks", "wait_weeks"], as_index=False).agg(
            median_trades=("trades", "median")
        )
        pivot_tr = hw_tr.pivot_table(
            index="hold_weeks",
            columns="wait_weeks",
            values="median_trades",
            aggfunc="mean",
        )
        if not pivot_tr.empty:
            plt.figure(figsize=(10, 6))
            plt.imshow(pivot_tr.values, aspect="auto")
            plt.colorbar(label="median trades")
            plt.title(f"Median trades by Hold/Wait for {strategy_name}")
            plt.xlabel("Wait Weeks")
            plt.ylabel("Hold Weeks")
            plt.xticks(
                range(len(pivot_tr.columns)), [str(int(c)) for c in pivot_tr.columns]
            )
            plt.yticks(
                range(len(pivot_tr.index)), [str(int(i)) for i in pivot_tr.index]
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(dir, f"{strategy_name}_heat_hold_wait_median_trades.png")
            )
            plt.close()

    # Best hold/wait cell (by median EAE)
    hw_rank = hw_med.sort_values("eae_median", ascending=False)
    if not hw_rank.empty:
        best = hw_rank.iloc[0]
        out_txt = os.path.join(dir, f"{strategy_name}_hold_wait_summary.txt")
        with open(out_txt, "w") as f:
            f.write("best_hold_wait_by_median_eae:\n")
            f.write(
                f"hold={int(best['hold_weeks'])}, "
                f"wait={int(best['wait_weeks'])}, "
                f"eae_median={best['eae_median']:.6f}, "
                f"eae_mean={best['eae_mean']:.6f}, "
                f"trades_median={best['trades_median']:.3f}\n\n"
            )
            f.write("top_25_hold_wait_cells_by_median_eae:\n")
            f.write(hw_rank.head(25).to_string(index=False) + "\n")


def run_return_mode_analysis(df: pd.DataFrame, dir: str, strategy_name: str):
    """Run return mode analysis including summary tables and boxplots.

    Args:
        df: DataFrame with strategy results
        dir: Directory for results
        strategy_name: Name of the strategy

    Returns:
        None
    """

    dir = os.path.join(dir, "return_mode_analysis")
    os.makedirs(dir, exist_ok=True)

    required_cols = ["return_mode", "eae"]
    for c in required_cols:
        if c not in df.columns:
            return

    d = df.copy()
    d["return_mode"] = as_numeric(d["return_mode"])
    d["eae"] = as_numeric(d["eae"])
    d = finite_frame(d, ["return_mode", "eae"])
    if d.empty:
        return

    # Summary table by return_mode
    tbl = (
        d.groupby("return_mode", as_index=False)
        .agg(
            configs=("eae", "count"),
            eae_mean=("eae", "mean"),
            eae_median=("eae", "median"),
            eae_std=("eae", "std"),
            trades_median=(
                ("trades", "median") if "trades" in d.columns else ("eae", "size")
            ),
            dd_mean=(
                ("max_drawdown", "mean")
                if "max_drawdown" in d.columns
                else ("eae", "mean")
            ),
        )
        .sort_values("return_mode")
    )

    out_txt = os.path.join(dir, f"{strategy_name}_return_mode_summary.txt")
    with open(out_txt, "w") as f:
        f.write("return_mode_summary:\n")
        f.write(tbl.to_string(index=False) + "\n")

    # Boxplot: EAE by return_mode
    modes = sorted(d["return_mode"].unique())
    data = []
    labels = []
    for m in modes:
        s = d.loc[d["return_mode"] == m, "eae"]
        s = as_numeric(s)
        s = s.dropna()
        if s.empty:
            continue
        data.append(s.to_numpy())
        labels.append(f"{int(m)}\n(n={len(s)}, med={s.median():.2f})")

    if len(data) > 0:
        plt.figure(figsize=(10, 6))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.title(f"eae by Return Mode for {strategy_name}")
        plt.xlabel("Return Mode")
        plt.ylabel("eae")
        plt.tight_layout()
        plt.savefig(os.path.join(dir, f"{strategy_name}_box_eae_by_return_mode.png"))
        plt.close()


def run_eae_analysis(df: pd.DataFrame, dir: str, strategy_name: str, top_k: int = 25):
    """Run EAE-focused analysis including scatter plots and top-k tables.

    Args:
        df: DataFrame with strategy results
        dir: Directory for results
        strategy_name: Name of the strategy
        top_k: Number of top configurations to report

    Returns:
        None
    """

    dir = os.path.join(dir, "eae_analysis")
    os.makedirs(dir, exist_ok=True)

    if "eae" not in df.columns:
        return

    d = df.copy()
    d["eae"] = as_numeric(d["eae"])
    d = finite_frame(d, ["eae"])
    if d.empty:
        return

    # Scatter: eae vs trades
    if "trades" in d.columns:
        x = as_numeric(d["trades"])
        y = as_numeric(d["eae"])
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m].to_numpy()
        y = y[m].to_numpy()
        if len(x) > 0:
            plt.figure(figsize=(7, 6))
            plt.scatter(x, y, alpha=0.35, s=14)
            plt.title(f"eae vs Trades for {strategy_name}")
            plt.xlabel("Trades")
            plt.ylabel("eae")
            plt.tight_layout()
            plt.savefig(os.path.join(dir, f"{strategy_name}_scatter_eae_vs_trades.png"))
            plt.close()

    # Scatter: eae vs |max_drawdown|
    if "max_drawdown" in d.columns:
        x = as_numeric(d["max_drawdown"]).abs()
        y = as_numeric(d["eae"])
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m].to_numpy()
        y = y[m].to_numpy()
        if len(x) > 0:
            plt.figure(figsize=(7, 6))
            plt.scatter(x, y, alpha=0.35, s=14)
            plt.title(f"eae vs |Max Drawdown| for {strategy_name}")
            plt.xlabel("|Max Drawdown|")
            plt.ylabel("eae")
            plt.tight_layout()
            plt.savefig(os.path.join(dir, f"{strategy_name}_scatter_eae_vs_dd_abs.png"))
            plt.close()

    # Top-k configs by EAE
    config_cols = [
        "bull_threshold",
        "bear_threshold",
        "return_mode",
        "wait_weeks",
        "hold_weeks",
        "analysis_mode",
        "strategy",
    ]
    keep_cols = [c for c in config_cols if c in d.columns]
    extra_cols = [
        c
        for c in [
            "trades",
            "cum_return_pct",
            "max_drawdown",
            "profit_factor",
            "win_rate",
            "eae",
        ]
        if c in d.columns
    ]
    cols = keep_cols + extra_cols

    top = d.sort_values("eae", ascending=False).head(top_k)
    out_txt = os.path.join(dir, f"{strategy_name}_top_{top_k}_by_eae.txt")
    with open(out_txt, "w") as f:
        f.write(f"top_{top_k}_by_eae:\n")
        f.write(top[cols].to_string(index=False) + "\n")


def analyze_timing_structure_stability(
    df: pd.DataFrame,
    dir: str,
    strategy_name: str,
    objective: str,
):
    """Analyze whether timing structure or thresholds drive performance.

    Args:
        df: DataFrame with strategy results
        dir: Directory for results
        strategy_name: Name of the strategy
        objective: Objective to analyze (e.g., "eae")

    Returns:
        None
    """

    dir = os.path.join(dir, "timing_structure_stability")
    os.makedirs(dir, exist_ok=True)

    for mode in sorted(df["return_mode"].dropna().unique()):
        sub = df[df["return_mode"] == mode]

        if sub.empty:
            continue

        med = (
            sub.groupby(["hold_weeks", "wait_weeks"], observed=True)[objective]
            .median()
            .reset_index(name="median_value")
        )

        best = (
            sub.groupby(["hold_weeks", "wait_weeks"], observed=True)[objective]
            .max()
            .reset_index(name="best_value")
        )

        for label, frame, col in [
            ("Median", med, "median_value"),
            ("Best", best, "best_value"),
        ]:
            pivot = frame.pivot(index="hold_weeks", columns="wait_weeks", values=col)

            plt.figure(figsize=(10, 6))
            plt.imshow(pivot.values, aspect="auto")
            plt.colorbar(label=objective)
            plt.title(f"{strategy_name} {label} {objective} (Return Mode {int(mode)})")
            plt.xlabel("Wait Weeks")
            plt.ylabel("Hold Weeks")
            plt.xticks(range(len(pivot.columns)), pivot.columns)
            plt.yticks(range(len(pivot.index)), pivot.index)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    dir, f"{strategy_name}_timing_{label.lower()}_mode_{int(mode)}.png"
                )
            )
            plt.close()


def analyze_local_parameter_robustness(
    df: pd.DataFrame,
    dir: str,
    strategy_name: str,
    objective: str,
    top_k: int = 1,
):
    """Evaluate local robustness around top configurations.

    Args:
        df: DataFrame with strategy results
        dir: Directory for results
        strategy_name: Name of the strategy
        objective: Objective to analyze (e.g., "eae")
        top_k: Number of top configurations to evaluate

    Returns:
        None
    """

    dir = os.path.join(dir, "local_parameter_robustness")
    os.makedirs(dir, exist_ok=True)

    top = df.sort_values(objective, ascending=False).head(top_k)

    rows = []

    for _, r in top.iterrows():
        mask = (
            (df["return_mode"] == r["return_mode"])
            & (df["hold_weeks"].between(r["hold_weeks"] - 1, r["hold_weeks"] + 1))
            & (df["wait_weeks"].between(r["wait_weeks"] - 1, r["wait_weeks"] + 1))
            & (
                df["bull_threshold"].between(
                    r["bull_threshold"] - 1, r["bull_threshold"] + 1
                )
            )
            & (
                df["bear_threshold"].between(
                    r["bear_threshold"] - 1, r["bear_threshold"] + 1
                )
            )
        )

        local = df[mask][objective].dropna()

        rows.append(
            {
                "center_value": r[objective],
                "local_mean": local.mean(),
                "local_std": local.std(),
                "local_min": local.min(),
                "local_max": local.max(),
                "count": len(local),
            }
        )

    pd.DataFrame(rows).to_csv(
        os.path.join(dir, f"{strategy_name}_local_robustness.csv"),
        index=False,
    )


def analyze_trade_count_dependence(
    df: pd.DataFrame,
    dir: str,
    strategy_name: str,
    objective: str,
):
    """Analyze relationship between trade count and performance.

    Args:
        df: DataFrame with strategy results
        dir: Directory for results
        strategy_name: Name of the strategy
        objective: Objective to analyze (e.g., "eae")

    Returns:
        None
    """

    dir = os.path.join(dir, "trade_count_dependence")
    os.makedirs(dir, exist_ok=True)

    if "trades" not in df.columns or objective not in df.columns:
        return

    d = df[["trades", objective]].copy()
    d["trades"] = as_numeric(d["trades"])
    d[objective] = as_numeric(d[objective])
    d = d.dropna()

    if d.empty:
        return

    d["trade_bucket"] = pd.qcut(d["trades"], 5, duplicates="drop")

    summary = (
        d.groupby("trade_bucket", observed=True)[objective]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    summary.to_csv(
        os.path.join(dir, f"{strategy_name}_trade_bucket_summary.csv"),
        index=False,
    )

    buckets = []
    labels = []
    for bucket, sub in d.groupby("trade_bucket", observed=True):
        vals = sub[objective].dropna().to_numpy()
        if len(vals) == 0:
            continue
        buckets.append(vals)
        labels.append(str(bucket))

    if len(buckets) == 0:
        return

    plt.figure(figsize=(10, 5))
    plt.boxplot(buckets, labels=labels, showfliers=False)
    plt.title(f"{strategy_name} {objective} By Trade Bucket")
    plt.xlabel("Trade Bucket")
    plt.ylabel(objective)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(dir, f"{strategy_name}_trade_bucket_boxplot.png"))
    plt.close()


def analyze_signal_quality_relationship(
    df: pd.DataFrame,
    dir: str,
    strategy_name: str,
    objective: str,
):
    """Analyze relationship between signal quality metrics and performance.

    Args:
        df: DataFrame with strategy results
        dir: Directory for results
        strategy_name: Name of the strategy
        objective: Objective to analyze (e.g., "eae")

    Returns:
        None
    """

    dir = os.path.join(dir, "signal_quality_relationship")
    os.makedirs(dir, exist_ok=True)

    pairs = [
        ("long_signal_ic", "Long Signal IC"),
        ("short_signal_ic", "Short Signal IC"),
        ("trades_ic", "Trades IC"),
    ]

    for col, label in pairs:
        if col not in df.columns:
            continue

        plt.figure(figsize=(7, 6))
        plt.scatter(df[col], df[objective], alpha=0.4)
        plt.title(f"{strategy_name} {objective} vs {label}")
        plt.xlabel(label)
        plt.ylabel(objective)
        plt.tight_layout()
        plt.savefig(os.path.join(dir, f"{strategy_name}_{col}_scatter.png"))
        plt.close()


def compare_long_short_vs_long_only(
    longshort_df: pd.DataFrame,
    long_df: pd.DataFrame,
    dir: str,
    objective: str,
):
    """Compare Long/Short and Long-only on matched configurations.

    Args:
        longshort_df: DataFrame with Long/Short strategy results
        long_df: DataFrame with Long-only strategy results
        dir: Directory for results
        objective: Objective to compare (e.g., "eae")

    Returns:
        None
    """

    dir = os.path.join(dir, "long_short_vs_long_only")
    os.makedirs(dir, exist_ok=True)

    keys = [
        "bull_threshold",
        "bear_threshold",
        "return_mode",
        "wait_weeks",
        "hold_weeks",
    ]

    merged = longshort_df.merge(
        long_df,
        on=keys,
        suffixes=("_ls", "_long"),
    )

    merged["delta"] = merged[f"{objective}_ls"] - merged[f"{objective}_long"]

    plt.figure(figsize=(10, 5))
    plt.hist(merged["delta"], bins=40)
    plt.axvline(merged["delta"].mean(), color="red", linestyle="--", label="Mean")
    plt.axvline(merged["delta"].median(), color="blue", linestyle="--", label="Median")
    plt.axvline(
        merged["delta"].quantile(0.25),
        color="green",
        linestyle="--",
        label="25th percentile",
    )
    plt.axvline(
        merged["delta"].quantile(0.75),
        color="green",
        linestyle="--",
        label="75th percentile",
    )
    plt.title("Delta Objective: Long/Short Minus Long-Only")
    plt.xlabel("Delta")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "delta_objective_histogram.png"))
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(
        merged[f"{objective}_long"],
        merged[f"{objective}_ls"],
        alpha=0.4,
    )
    lo = min(merged[f"{objective}_long"].min(), merged[f"{objective}_ls"].min())
    hi = max(merged[f"{objective}_long"].max(), merged[f"{objective}_ls"].max())
    plt.plot([lo, hi], [lo, hi])
    plt.title("Objective: Long-Only vs Long/Short")
    plt.xlabel("Long-Only")
    plt.ylabel("Long/Short")
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "objective_scatter_ls_vs_long.png"))
    plt.close()


def analyze_null_dominance(
    df: pd.DataFrame,
    null_df: pd.DataFrame,
    dir: str,
    strategy_name: str,
    objective: str,
    top_k: int,
):
    """Analyze dominance of real performance over null.

    Args:
        df: DataFrame with real strategy results
        null_df: DataFrame with null strategy results
        dir: Directory for results
        strategy_name: Name of the strategy
        objective: Objective to analyze (e.g., "eae")
        top_k: Number of top configurations to analyze

    Returns:
        None
    """

    dir = os.path.join(dir, "null_dominance")
    os.makedirs(dir, exist_ok=True)

    merged = df.merge(
        null_df,
        on=[
            "bull_threshold",
            "bear_threshold",
            "return_mode",
            "wait_weeks",
            "hold_weeks",
            "strategy",
        ],
        suffixes=("", "_null"),
    )

    merged["z"] = (merged[objective] - merged[f"{objective}_null"]) / merged[
        f"{objective}_null"
    ].std()

    plt.figure(figsize=(10, 5))
    plt.hist(merged["z"].dropna(), bins=40)
    plt.axvline(merged["z"].mean(), color="red", linestyle="--", label="Mean")
    plt.axvline(merged["z"].median(), color="blue", linestyle="--", label="Median")
    plt.axvline(
        merged["z"].quantile(0.25),
        color="green",
        linestyle="--",
        label="25th percentile",
    )
    plt.axvline(
        merged["z"].quantile(0.75),
        color="green",
        linestyle="--",
        label="75th percentile",
    )
    plt.title(f"{strategy_name} Z Score vs Null")
    plt.xlabel("Z Score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir, f"{strategy_name}_zscore_hist.png"))
    plt.close()

    merged.sort_values(objective, ascending=False).head(top_k).to_csv(
        os.path.join(dir, f"{strategy_name}_top_with_null.csv"),
        index=False,
    )


def analyze_multi_objective_tradeoffs(
    df: pd.DataFrame,
    dir: str,
    strategy_name: str,
    objective: str,
):
    """Analyze tradeoffs between performance and risk metrics.

    Args:
        df: DataFrame with strategy results
        dir: Directory for results
        strategy_name: Name of the strategy
        objective: Objective to analyze (e.g., "eae")

    Returns:
        None
    """

    dir = os.path.join(dir, "multi_objective_tradeoffs")
    os.makedirs(dir, exist_ok=True)

    pairs = [
        ("max_drawdown", "Max Drawdown"),
        ("trades", "Trades"),
        ("profit_factor", "Profit Factor"),
        ("win_rate", "Win Rate"),
    ]

    for col, label in pairs:
        if col not in df.columns:
            continue

        plt.figure(figsize=(7, 6))
        plt.scatter(df[col], df[objective], alpha=0.4)
        plt.title(f"{strategy_name} {objective} vs {label}")
        plt.xlabel(label)
        plt.ylabel(objective)
        plt.tight_layout()
        plt.savefig(os.path.join(dir, f"{strategy_name}_{col}_tradeoff.png"))
        plt.close()


def build_deployment_shortlist(
    df: pd.DataFrame,
    dir: str,
    strategy_name: str,
    objective: str,
    top_n: int = 3,
):
    """Create a deployment-ready shortlist of configurations.

    Args:
        df: DataFrame with strategy results
        dir: Directory for results
        strategy_name: Name of the strategy
        objective: Objective to analyze (e.g., "eae")
        top_n: Number of top configurations to include

    Returns:
        None
    """

    dir = os.path.join(dir, "deployment_shortlist")
    os.makedirs(dir, exist_ok=True)

    shortlist = df[df["trades"] > 5].sort_values(objective, ascending=False).head(top_n)

    shortlist.to_csv(
        os.path.join(dir, f"{strategy_name}_deployment_shortlist.csv"),
        index=False,
    )


def pareto_front(
    df: pd.DataFrame,
    dir: str,
    strategy_name: str,
    maximize_cols: list[str],
    minimize_cols: list[str],
    keep_cols: Optional[list[str]] = None,
    out_name: str = "pareto_front",
):
    """Compute a Pareto front over selected metrics and save artifacts.

    Args:
        df: DataFrame of strategy results
        dir: Directory for results
        strategy_name: Name of the strategy
        maximize_cols: Columns to maximize
        minimize_cols: Columns to minimize
        keep_cols: Optional columns to include in output
        out_name: Output file prefix

    Returns:
        pd.DataFrame of Pareto-optimal rows
    """

    dir = os.path.join(dir, "pareto_front")
    os.makedirs(dir, exist_ok=True)

    cols = list(maximize_cols) + list(minimize_cols)
    missing = [c for c in cols if c not in df.columns]
    if len(missing) > 0:
        return pd.DataFrame()

    d = df.copy()

    for c in cols:
        d[c] = as_numeric(d[c])

    d = finite_frame(d, cols)
    if d.empty:
        return pd.DataFrame()

    score = pd.DataFrame(index=d.index)

    for c in maximize_cols:
        score[c] = d[c]

    for c in minimize_cols:
        if c == "max_drawdown":
            score[c] = -d[c].abs()
        else:
            score[c] = -d[c]

    x = score.to_numpy()
    n = x.shape[0]
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        better_or_equal = np.all(x >= x[i], axis=1)
        strictly_better = np.any(x > x[i], axis=1)
        dominates_i = better_or_equal & strictly_better
        dominates_i[i] = False
        if np.any(dominates_i):
            is_pareto[i] = False

    front = d.iloc[np.where(is_pareto)[0]].copy()

    if keep_cols is None:
        config_cols = [
            "bull_threshold",
            "bear_threshold",
            "return_mode",
            "wait_weeks",
            "hold_weeks",
            "strategy",
        ]
        keep_cols = [c for c in config_cols if c in front.columns]

    out_cols = keep_cols + cols
    out_cols = [c for c in out_cols if c in front.columns]

    if len(maximize_cols) > 0:
        front = front.sort_values(maximize_cols[0], ascending=False)

    front.to_csv(
        os.path.join(dir, f"{strategy_name}_{out_name}.csv"),
        index=False,
        columns=out_cols,
    )

    if len(cols) == 2:
        a = cols[0]
        b = cols[1]

        plt.figure(figsize=(7, 6))
        plt.scatter(d[a], d[b], alpha=0.25, s=12, label="All")
        plt.scatter(front[a], front[b], alpha=0.9, s=18, label="Pareto")
        plt.title(f"Pareto Front For {strategy_name}")
        plt.xlabel(a)
        plt.ylabel(b)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(dir, f"{strategy_name}_{out_name}_scatter.png"))
        plt.close()

    return front


def as_numeric(s: pd.Series) -> pd.Series:
    """Coerce to numeric and drop +/-inf (keeps NaN)."""
    s = pd.to_numeric(s, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)


def finite_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a 1D numeric vector with only finite values."""
    if col not in df.columns:
        return pd.Series(dtype=float)
    s = as_numeric(df[col])
    return s.dropna()


def finite_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return df with only finite values in the specified columns."""
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return df.iloc[0:0].copy()
    d = df.copy()
    for c in keep:
        d[c] = as_numeric(d[c])
    return d.dropna(subset=keep)
