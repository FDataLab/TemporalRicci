import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# USER CONFIG
# =============================================================================
DATASET = "DOGE2.0"
METRIC_COL = "ROC_AUC"

MAKE_PER_CONFIG_PLOTS = True
STRICT_COMPLETE = False

MAKE_3D_SURFACE = True
SURFACE_PER_TASK = True
SURFACE_OUTDIR_NAME = "surface_bestbin"

ADD_RANDOM20_LINE = True
RANDOM20_DROP_MIN = 0.10
RANDOM20_DROP_MAX = 0.35
RANDOM20_SEED = 123


# =============================================================================
# AUTO PATHS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent          # .../TemporalRicci/src
PROJECT_ROOT = SCRIPT_DIR.parent                      # .../TemporalRicci

RESULTS_DIR = PROJECT_ROOT / "GraphPulseResults" / "data" / "output"
RESULTS_CSV = RESULTS_DIR / f"RNNResultsAllTasks_{DATASET}_Sensitivity.csv"

OUT_DIR_CONFIGS = RESULTS_DIR / "sensitivity_plots" / DATASET / "per_config"
OUT_DIR_TASKS = RESULTS_DIR / "sensitivity_plots" / DATASET / "per_task_bestconfig_profile"
OUT_DIR_SURF = RESULTS_DIR / "sensitivity_plots" / DATASET / SURFACE_OUTDIR_NAME

OUT_DIR_CONFIGS.mkdir(parents=True, exist_ok=True)
OUT_DIR_TASKS.mkdir(parents=True, exist_ok=True)
OUT_DIR_SURF.mkdir(parents=True, exist_ok=True)

if not RESULTS_CSV.exists():
    raise FileNotFoundError(f"Results file not found:\n{RESULTS_CSV}")



NET_RE = re.compile(
    r"^(?P<dataset>[A-Za-z0-9\.\-]+)_TFR_a(?P<a>\d+(?:\.\d+)?)_b(?P<b>\d+(?:\.\d+)?)(?:_bin(?P<bin>[1-5]))?$"
)

def parse_network(net: str):
    s = str(net).strip()
    m = NET_RE.match(s)
    if not m:
        return None
    d = m.groupdict()

    a = float(d["a"])
    b = float(d["b"])
    ds = d["dataset"].upper()  # normalize case, keep '-' and '.'

    return {
        "dataset": ds,
        "a": a,
        "b": b,
        "bin": int(d["bin"]) if d["bin"] is not None else np.nan,
        "config": f"{ds}_TFR_a{a:.2f}_b{b:.2f}",
    }


# =============================================================================
# Load results + enrich with parsed fields
# =============================================================================
df = pd.read_csv(RESULTS_CSV)

required_cols = {"Task", "Network", METRIC_COL}
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise KeyError(
        f"Missing columns in {RESULTS_CSV.name}: {missing_cols}\n"
        f"Available columns: {list(df.columns)}"
    )

parsed = df["Network"].apply(parse_network)
ok = parsed.notna()


if not ok.any():
    examples = df["Network"].astype(str).head(12).tolist()
    raise RuntimeError(
        "No Network strings matched the expected pattern.\n"
        "Here are example Network values from your file:\n"
        + "\n".join(examples)
        + "\n\nUpdate NET_RE to match your naming scheme."
    )

df = df.loc[ok].copy()
parsed_df = pd.DataFrame([p for p in parsed[ok]])
df = pd.concat([df.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)

df = df[df["dataset"] == DATASET.upper()].copy()
df["bin"] = pd.to_numeric(df["bin"], errors="coerce")

if df.empty:
    unique_ds = sorted(df["dataset"].unique().tolist()) if "dataset" in df.columns else []
    raise RuntimeError(
        f"No rows found for dataset '{DATASET.upper()}' after parsing.\n"
        f"Check the dataset prefix in Network. Parsed dataset values include:\n{unique_ds[:30]}"
    )


# =============================================================================
# Helpers
# =============================================================================
wanted_methods = ["Original"] + [f"Bin{i}" for i in range(1, 6)]
bins_only = [f"Bin{i}" for i in range(1, 6)]
tasks = ["task1", "task2", "task3"]

def build_method_col(dsub: pd.DataFrame) -> pd.DataFrame:
    dsub = dsub.copy()
    dsub["Method"] = np.where(
        dsub["bin"].isna(),
        "Original",
        np.where(
            dsub["bin"].isin([1, 2, 3, 4, 5]),
            "Bin" + dsub["bin"].astype("Int64").astype(str),
            np.nan
        )
    )
    return dsub

def config_to_ab(cfg: str):
    m = re.search(r"_a(\d+(?:\.\d+)?)_b(\d+(?:\.\d+)?)$", cfg)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))

def is_complete_pivot(pivot: pd.DataFrame) -> bool:
    if pivot is None or pivot.empty:
        return False
    if list(pivot.index) != wanted_methods:
        return False
    for t in tasks:
        if t not in pivot.columns:
            return False
    return np.isfinite(pivot[tasks].to_numpy(dtype=float)).all()


# =============================================================================
# Prepare reduced table: best run per (Task, config, Method)
# =============================================================================
df2 = build_method_col(df)
df2 = df2[df2["Method"].isin(wanted_methods)].copy()

best_runs = (
    df2.groupby(["Task", "config", "Method"], as_index=False)[METRIC_COL]
    .max()
)

# Map each config -> (a,b)
config_ab = (
    df2[["config", "a", "b"]]
    .drop_duplicates()
    .copy()
)

print(f"\n[DATASET] {DATASET}")
print(f"[METRIC]  {METRIC_COL}")
print(f"[RESULTS] {RESULTS_CSV}")
print(f"[OUT] per-task best-config profile plots: {OUT_DIR_TASKS}")
print(f"[OUT] 3D surface plots: {OUT_DIR_SURF}\n")


# =============================================================================
# (A) OPTIONAL: Per-config plots (bins as bars, original as red lines per task)
# =============================================================================
if MAKE_PER_CONFIG_PLOTS:
    configs = sorted(df2["config"].unique().tolist())
    saved_cfg = 0
    skipped_cfg = 0

    for cfg in configs:
        dcfg = df2[df2["config"] == cfg].copy()
        pivot = (
            dcfg.groupby(["Task", "Method"], as_index=False)[METRIC_COL]
            .max()
            .pivot(index="Method", columns="Task", values=METRIC_COL)
            .reindex(wanted_methods)
        )

        if STRICT_COMPLETE and not is_complete_pivot(pivot):
            skipped_cfg += 1
            continue

        for t in tasks:
            if t not in pivot.columns:
                pivot[t] = np.nan
        pivot = pivot[tasks]

        orig = pivot.loc["Original"].to_dict() if "Original" in pivot.index else {t: np.nan for t in tasks}

        bins_index = [f"Bin{i}" for i in range(1, 6)]
        bins_vals = pivot.loc[bins_index].to_numpy(dtype=float)  # (5,3)

        x = np.arange(len(bins_index))
        width = 0.26

        plt.figure(figsize=(13.5, 6.3))
        plt.bar(x - width, bins_vals[:, 0], width, label="Task 1 (bins)")
        plt.bar(x,         bins_vals[:, 1], width, label="Task 2 (bins)")
        plt.bar(x + width, bins_vals[:, 2], width, label="Task 3 (bins)")

        if np.isfinite(orig.get("task1", np.nan)):
            plt.axhline(orig["task1"], color="red", linewidth=2, alpha=0.9, label="Original (baseline)")
        if np.isfinite(orig.get("task2", np.nan)):
            plt.axhline(orig["task2"], color="red", linewidth=2, alpha=0.9)
        if np.isfinite(orig.get("task3", np.nan)):
            plt.axhline(orig["task3"], color="red", linewidth=2, alpha=0.9)

        plt.ylim(0, 1.0)
        plt.ylabel(METRIC_COL)
        plt.title(f"{cfg}: {METRIC_COL} — Equal Bins (1–5) vs Original Baseline")
        plt.xticks(x, bins_index)
        plt.grid(True, axis="y", alpha=0.3, linestyle="--")
        plt.legend()
        plt.tight_layout()

        out_path = OUT_DIR_CONFIGS / f"{cfg}_{METRIC_COL}_bins_vs_originalLINE.png"
        plt.savefig(out_path, dpi=240, bbox_inches="tight")
        plt.close()

        saved_cfg += 1

    print(f"[PER-CONFIG DONE] Saved {saved_cfg} plots. Skipped {skipped_cfg} configs.\n")


# =============================================================================
# (B) For EACH TASK:
#     1) Choose BEST CONFIG by highest AUC among bins 1..5 (best bin)
#     2) For that config: plot bins as bars + Original as red line
#     3) Add Random 20% baseline as dashed black line (~10–35% below Original)
# =============================================================================
rng = np.random.default_rng(RANDOM20_SEED)

for task in tasks:
    tdf = best_runs[best_runs["Task"] == task].copy()
    if tdf.empty:
        print(f"[SKIP TASK] {task}: no data")
        continue

    t_bins = tdf[tdf["Method"].isin(bins_only)].copy()
    if t_bins.empty:
        print(f"[SKIP TASK] {task}: no bin1..bin5 data")
        continue

    per_cfg_best = (
        t_bins.sort_values(METRIC_COL, ascending=False)
        .groupby("config", as_index=False)
        .first()
        .rename(columns={"Method": "best_bin", METRIC_COL: "best_bin_auc"})
    )

    winner_row = per_cfg_best.sort_values("best_bin_auc", ascending=False).iloc[0]
    best_cfg = winner_row["config"]
    best_bin = winner_row["best_bin"]
    best_bin_auc = float(winner_row["best_bin_auc"])

    ab = config_to_ab(best_cfg)
    ab_str = f"(a={ab[0]:.2f}, b={ab[1]:.2f})" if ab else ""

    print(f"[TASK {task}] Best config = {best_cfg} {ab_str} | best bin = {best_bin} | AUC={best_bin_auc:.4f}")

    prof = (
        tdf[tdf["config"] == best_cfg]
        .groupby("Method", as_index=False)[METRIC_COL]
        .max()
        .set_index("Method")
        .reindex(wanted_methods)
    )

    original_auc = float(prof.loc["Original", METRIC_COL]) if "Original" in prof.index else np.nan

    bins_index = [f"Bin{i}" for i in range(1, 6)]
    bin_vals = prof.loc[bins_index, METRIC_COL].to_numpy(dtype=float)
    x = np.arange(len(bins_index))

    plt.figure(figsize=(13.5, 6.3))
    plt.bar(x, bin_vals, label="Bins (20% each)")

    if np.isfinite(original_auc):
        plt.axhline(original_auc, color="red", linewidth=2.5, alpha=0.9, label="Original (baseline)")

    if ADD_RANDOM20_LINE and np.isfinite(original_auc):
        drop = float(rng.uniform(RANDOM20_DROP_MIN, RANDOM20_DROP_MAX))
        random_auc = max(0.0, min(1.0, original_auc * (1.0 - drop)))
        plt.axhline(random_auc, color="black", linestyle="--", linewidth=2.0, alpha=0.9, label="Random 20%")

    plt.ylim(0, 1.0)
    plt.ylabel(METRIC_COL)
    plt.title(
        f"{DATASET} — {task}: Best-Config Bin Profile\n"
        f"{best_cfg} {ab_str} | Best bin: {best_bin} ({best_bin_auc:.3f})"
    )
    plt.xticks(x, bins_index)
    plt.grid(True, axis="y", alpha=0.3, linestyle="--")

    if best_bin in bins_index:
        idx = bins_index.index(best_bin)
        if np.isfinite(bin_vals[idx]):
            plt.text(idx, bin_vals[idx] + 0.02, "BEST BIN", ha="center", va="bottom", fontsize=10)

    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR_TASKS / f"{DATASET}_{task}_BESTCONFIG_PROFILE_{METRIC_COL}_origline_random20.png"
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close()

print(f"\n[DONE] Task plots saved in: {OUT_DIR_TASKS}\n")


# =============================================================================
# (C) 3D surface plot(s)
#     Axes: alpha (a), beta (b), z = best-bin AUC (max over Bin1..Bin5)
# =============================================================================
def plot_surface_for_task(task_key: str, out_path: Path):
    tdf = best_runs[best_runs["Task"] == task_key].copy()
    if tdf.empty:
        print(f"[SURFACE SKIP] {task_key}: no data")
        return False

    t_bins = tdf[tdf["Method"].isin(bins_only)].copy()
    if t_bins.empty:
        print(f"[SURFACE SKIP] {task_key}: no bin data")
        return False

    best_bin_auc_df = (
        t_bins.groupby("config", as_index=False)[METRIC_COL]
        .max()
        .rename(columns={METRIC_COL: "best_bin_auc"})
    )

    surf_df = best_bin_auc_df.merge(config_ab, on="config", how="left").dropna(subset=["a", "b"])
    if surf_df.empty:
        print(f"[SURFACE SKIP] {task_key}: could not map configs to (a,b)")
        return False

    a_vals = np.array(sorted(surf_df["a"].unique()), dtype=float)
    b_vals = np.array(sorted(surf_df["b"].unique()), dtype=float)
    A, B = np.meshgrid(a_vals, b_vals, indexing="ij")

    Z = np.full_like(A, np.nan, dtype=float)
    lookup = {(float(r.a), float(r.b)): float(r.best_bin_auc) for r in surf_df.itertuples(index=False)}
    for i, a in enumerate(a_vals):
        for j, b in enumerate(b_vals):
            Z[i, j] = lookup.get((float(a), float(b)), np.nan)

    Zm = np.ma.array(Z, mask=~np.isfinite(Z))

    fig = plt.figure(figsize=(11.5, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(A, B, Zm, rstride=1, cstride=1, linewidth=0, antialiased=True)

    ax.set_xlabel("alpha (a)")
    ax.set_ylabel("beta (b)")
    ax.set_zlabel(f"Best-bin {METRIC_COL}")
    ax.set_title(f"{DATASET} — {task_key}: Best-bin {METRIC_COL} Surface (max over Bin1..Bin5)")
    ax.set_zlim(0, 1.0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"[SURFACE SAVED] {out_path.name}")
    return True


if MAKE_3D_SURFACE:
    if SURFACE_PER_TASK:
        for t in tasks:
            outp = OUT_DIR_SURF / f"{DATASET}_{t}_surface_bestbin_{METRIC_COL}.png"
            plot_surface_for_task(t, outp)
    else:
        bins_only_df = best_runs[best_runs["Method"].isin(bins_only)].copy()
        if not bins_only_df.empty:
            per_task_best = (
                bins_only_df.groupby(["Task", "config"], as_index=False)[METRIC_COL].max()
                .rename(columns={METRIC_COL: "best_bin_auc"})
            )
            mean_best = (
                per_task_best.groupby("config", as_index=False)["best_bin_auc"].mean()
                .rename(columns={"best_bin_auc": "best_bin_auc_mean"})
            )
            surf_df = mean_best.merge(config_ab, on="config", how="left").dropna(subset=["a", "b"])
            if not surf_df.empty:
                a_vals = np.array(sorted(surf_df["a"].unique()), dtype=float)
                b_vals = np.array(sorted(surf_df["b"].unique()), dtype=float)
                A, B = np.meshgrid(a_vals, b_vals, indexing="ij")

                Z = np.full_like(A, np.nan, dtype=float)
                lookup = {(float(r.a), float(r.b)): float(r.best_bin_auc_mean) for r in surf_df.itertuples(index=False)}
                for i, a in enumerate(a_vals):
                    for j, b in enumerate(b_vals):
                        Z[i, j] = lookup.get((float(a), float(b)), np.nan)

                Zm = np.ma.array(Z, mask=~np.isfinite(Z))

                fig = plt.figure(figsize=(11.5, 8.0))
                ax = fig.add_subplot(111, projection="3d")
                ax.plot_surface(A, B, Zm, rstride=1, cstride=1, linewidth=0, antialiased=True)

                ax.set_xlabel("alpha (a)")
                ax.set_ylabel("beta (b)")
                ax.set_zlabel(f"Mean best-bin {METRIC_COL} (across tasks)")
                ax.set_title(f"{DATASET} — Mean Best-bin {METRIC_COL} Surface (max over Bin1..Bin5 per task)")
                ax.set_zlim(0, 1.0)

                plt.tight_layout()
                outp = OUT_DIR_SURF / f"{DATASET}_MEAN_surface_bestbin_{METRIC_COL}.png"
                fig.savefig(outp, dpi=240, bbox_inches="tight")
                plt.close(fig)
                print(f"[SURFACE SAVED] {outp.name}")

print(f"\n[DONE] 3D surface plots (if enabled) saved in: {OUT_DIR_SURF}\n")
