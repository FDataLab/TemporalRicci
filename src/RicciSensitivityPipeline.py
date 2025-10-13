import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

from normalization import normalize_values
from temporalRicci import compute_forman_ricci

# =========================
# Config
# =========================
DATASET = "BEPRO.csv"
DATA_DIR = "../data"

OUT_DIR = Path.cwd() / "RicciResults" / "ricci_values" / "BEPRO"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_NROWS = None  # None = full dataset

# α, β evenly spaced, (equal step)
ALPHA_START, ALPHA_END, ALPHA_STEP = 3, 3, 1
BETA_START,  BETA_END,  BETA_STEP  = 1, 1, 1
ALPHAS = np.round(np.arange(ALPHA_START, ALPHA_END + 1e-12, ALPHA_STEP), 2).tolist()
BETAS  = np.round(np.arange(BETA_START,  BETA_END  + 1e-12, BETA_STEP),  2).tolist()

print("=== CONFIG ===")
print(f"CWD     : {Path.cwd().resolve()}")
print(f"DATA_DIR: {Path(DATA_DIR).resolve()}")
print(f"DATASET : {DATASET}")
print(f"OUT_DIR : {OUT_DIR.resolve()}")
print(f"ALPHAS ({len(ALPHAS)}): {ALPHAS}")
print(f"BETAS  ({len(BETAS)}):  {BETAS}\n")

# =========================
# Load & Graph
# =========================
csv_path = Path(DATA_DIR) / DATASET
df = pd.read_csv(csv_path, nrows=TEST_NROWS)
required = {"from","to","timestamp","value"}
if not required.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required}")

df["value"] = pd.to_numeric(df["value"], errors="coerce")
df["value"] = normalize_values(df["value"], 10)

G = nx.from_pandas_edgelist(
    df, source="from", target="to",
    edge_attr=["value","timestamp"],
    create_using=nx.MultiDiGraph()
)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

# =========================
# Helpers
# =========================
def save_hist(series, title, path):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(series, bins=50, edgecolor="black")
    ax.set_title(title); ax.set_xlabel("Temporal Ricci value"); ax.set_ylabel("Frequency")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def tailaware_bins_10(values: np.ndarray) -> pd.Series:
    s = pd.Series(values.astype(float))
    pcts = [0.00, 0.50, 0.75, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 1.00]

    qs = s.quantile(pcts, interpolation="linear").to_numpy()
    edges = np.unique(qs)
    if len(edges) < 11:
        ranks = s.rank(method="first") / (len(s)+1.0)
        return pd.qcut(ranks, q=10, labels=list(range(1,11))).astype(int)
    return pd.cut(s, bins=edges, labels=list(range(1,11)), include_lowest=True).astype(int)

def bin_pairwise_and_edgewise(ricci_df: pd.DataFrame):
    """
    Returns:
      ricci_df_binned: original df with 'bin_edge' (edge-wise) and 'bin_pair' (pair-wise)
      edge_counts (per bin 1..10)
      node_counts_pairwise (unique nodes per bin using pair-wise bins)
      pairs_df: DataFrame with ['from','to','pair_value','bin_pair']
    """
    vals = pd.to_numeric(ricci_df["Temporal Ricci value"], errors="coerce").fillna(0.0)

    # Edge-wise bins (directly on edge values)
    bins_edge = tailaware_bins_10(vals.to_numpy())
    ricci_df = ricci_df.copy()
    ricci_df["bin_edge"] = bins_edge

    # Pair-wise aggregation (median per (from,to)), then bin
    pairs = (
        ricci_df.groupby(["from","to"], as_index=False)["Temporal Ricci value"]
        .median()
        .rename(columns={"Temporal Ricci value":"pair_value"})
    )
    pairs["bin_pair"] = tailaware_bins_10(pairs["pair_value"].to_numpy())

    # Map pair-wise bins back to edges (for reporting both views)
    ricci_df = ricci_df.merge(pairs[["from","to","bin_pair"]], on=["from","to"], how="left")

    # Edge counts per edge-wise bin
    bin_ids = list(range(1,11))
    edge_counts = {b: int((ricci_df["bin_edge"]==b).sum()) for b in bin_ids}

    # Unique node counts per pair-wise bin
    node_counts_pair = {}
    for b in bin_ids:
        sub = pairs[pairs["bin_pair"]==b]
        nodes = pd.unique(pd.concat([sub["from"], sub["to"]], ignore_index=True))
        node_counts_pair[b] = int(len(nodes))

    return ricci_df, edge_counts, node_counts_pair, pairs

def entropy_effective_bins(pair_bins: pd.Series) -> float:
    """Effective number of bins = exp(H), H in nats over 10 bins."""
    counts = pair_bins.value_counts().reindex(range(1,11), fill_value=0).to_numpy()
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    H = -np.sum(p[p>0]*np.log(p[p>0]))
    return float(np.exp(H))

def separation_score(pairs_df: pd.DataFrame) -> float:
    """Avg adjacent-bin separation: (median_{b+1}-median_b) / (pooled IQR + eps)."""
    eps = 1e-12
    med = {}; iqr = {}
    for b in range(1,11):
        v = pairs_df.loc[pairs_df["bin_pair"]==b, "pair_value"]
        if len(v)==0:
            med[b] = np.nan; iqr[b] = np.nan
        else:
            med[b] = float(np.median(v))
            q75, q25 = np.percentile(v, 75), np.percentile(v, 25)
            iqr[b] = float(q75 - q25)
    scores = []
    for b in range(1,10):
        if np.isfinite(med[b]) and np.isfinite(med[b+1]):
            denom = ((iqr[b] if np.isfinite(iqr[b]) else 0.0) + (iqr[b+1] if np.isfinite(iqr[b+1]) else 0.0))/2.0
            scores.append( (med[b+1]-med[b]) / (denom + eps) )
    return float(np.nanmean(scores)) if len(scores) else 0.0

# metrics helpers
def gini_from_counts(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(float))
    s = x.sum()
    if s <= 0:
        return 0.0
    n = len(x)
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * (cumx / s).sum()) / n)

def trimmed_mean_from_counts(counts: np.ndarray, trim: float = 0.2) -> float:
    k = len(counts)
    t = int(np.floor(trim * k))
    x = np.sort(counts.astype(float))
    if 2 * t >= k:
        return float(np.mean(x))
    return float(np.mean(x[t: k - t]))

def node_dominant_bins(pairs_df: pd.DataFrame) -> pd.Series:
    """Return Series: index=node, value=dominant bin (mode; ties -> median)."""
    a = pairs_df[['from', 'bin_pair']].rename(columns={'from': 'node'})
    b = pairs_df[['to',   'bin_pair']].rename(columns={'to':   'node'})
    nb = pd.concat([a, b], ignore_index=True)
    def _mode_or_median(s):
        vc = s.value_counts()
        top = vc[vc == vc.max()].index.to_list()
        if len(top) == 1:
            return int(top[0])
        return int(np.median(top))  # tie-break
    return nb.groupby('node')['bin_pair'].agg(_mode_or_median)

def pielou_evenness_from_node_bins(node_bins: pd.Series) -> float:
    """Pielou's J on dominant-bin node histogram (unique assignment)."""
    counts = node_bins.value_counts().reindex(range(1,11), fill_value=0).to_numpy()
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    H = -np.sum(p[p>0] * np.log(p[p>0]))
    return float(H / np.log(10.0))

# =========================
# Sweep α,β → save Ricci, bins, and sensitivity
# =========================
records_edge_counts = []   # edge counts per bin (edge-wise)
records_node_counts = []   # node counts per bin (pair-wise)
records_sens_rows   = []   # one row per (alpha,beta) with all metrics

total_jobs = len(ALPHAS)*len(BETAS)
print(f"Total (alpha,beta) runs: {total_jobs}\n")
job = 0

for a in ALPHAS:
    for b in BETAS:
        job += 1
        print(f"[{job}/{total_jobs}] alpha={a:.2f}, beta={b:.2f}")

        # skip recompute if CSV exists
        csv_name = f"BEPRO_TFR_a{a:.2f}_b{b:.2f}.csv"
        out_csv = OUT_DIR / csv_name
        if out_csv.exists():
            print(f"  -> CSV exists, loading and skipping recompute: {out_csv.resolve()}")
            ricci_df = pd.read_csv(out_csv)
        else:
            try:
                ricci_df = compute_forman_ricci(G, alpha=float(a), beta=float(b))
            except Exception as e:
                print("  compute_forman_ricci failed:", e)
                continue
            ricci_df.to_csv(out_csv, index=False)
            print("  -> CSV written:", out_csv.resolve())

        hist_path = OUT_DIR / f"BEPRO_TFR_a{a:.2f}_b{b:.2f}_hist.png"
        save_hist(ricci_df["Temporal Ricci value"], f"BEPRO TFR (a={a:.2f}, b={b:.2f})", hist_path)
        print("  -> HIST:", hist_path.resolve())

        # Binning
        ricci_binned, edge_counts, node_counts_pair, pairs = bin_pairwise_and_edgewise(ricci_df)

        for i in range(1, 11):
            bin_edges = ricci_binned[ricci_binned["bin_edge"] == i]
            if not bin_edges.empty:
                out_bin_file = OUT_DIR / f"{Path(csv_name).stem}_bin{i}.csv"
                bin_edges.to_csv(out_bin_file, index=False)
                print(f"  -> BIN FILE: {out_bin_file.resolve()}")

        # Save per-file bin counts
        out_bins_csv = OUT_DIR / f"{Path(csv_name).stem}_bin_counts.csv"
        per_file = pd.DataFrame({
            "bin": list(range(1,11)),
            "edge_count_edgewise": [edge_counts[i] for i in range(1,11)],
            "node_count_pairwise": [node_counts_pair[i] for i in range(1,11)],
        })
        per_file.to_csv(out_bins_csv, index=False)
        print("  -> BIN COUNTS:", out_bins_csv.resolve())

        # Combined tables for counts
        for i in range(1,11):
            records_edge_counts.append({
                "alpha": float(a), "beta": float(b),
                "file": csv_name, "bin": i,
                "edge_count_edgewise": edge_counts[i]
            })
            records_node_counts.append({
                "alpha": float(a), "beta": float(b),
                "file": csv_name, "bin": i,
                "node_count_pairwise": node_counts_pair[i]
            })

        # Node-focused metrics
        bin_ids = list(range(1,11))
        nodes_per_bin = np.array([node_counts_pair[i] for i in bin_ids], dtype=float)
        mean_nodes_per_bin = float(np.mean(nodes_per_bin))
        all_nodes_total = int(len(pd.unique(pd.concat([pairs["from"], pairs["to"]], ignore_index=True))))
        coverage_rate = (mean_nodes_per_bin / max(all_nodes_total, 1)) if all_nodes_total else 0.0
        eff_bins = entropy_effective_bins(pairs["bin_pair"])
        sep_score = separation_score(pairs)
        median_nodes_per_bin = float(np.median(nodes_per_bin))
        trimmed_mean_nodes_per_bin = trimmed_mean_from_counts(nodes_per_bin, trim=0.2)
        node_bins_series = node_dominant_bins(pairs)
        node_evenness = pielou_evenness_from_node_bins(node_bins_series)
        node_gini = gini_from_counts(
            node_bins_series.value_counts().reindex(range(1,11), fill_value=0).to_numpy()
        )

        # Edge-focused metrics (NEW)
        edges_per_bin = np.array([edge_counts[i] for i in range(1, 11)], dtype=float)
        mean_edges_per_bin = float(np.mean(edges_per_bin))
        median_edges_per_bin = float(np.median(edges_per_bin))
        total_unique_edges = int(len(ricci_df))
        edge_coverage_rate = (mean_edges_per_bin / max(total_unique_edges, 1)) if total_unique_edges else 0.0

        records_sens_rows.append({
            "alpha": float(a),
            "beta": float(b),

            "coverage_rate": coverage_rate,
            "mean_nodes_per_bin": mean_nodes_per_bin,
            "effective_bins": eff_bins,
            "separation_score": sep_score,
            "median_nodes_per_bin": median_nodes_per_bin,
            "trimmed_mean_nodes_per_bin": trimmed_mean_nodes_per_bin,
            "node_evenness": node_evenness,
            "node_gini": node_gini,

            "edge_coverage_rate": edge_coverage_rate,
            "mean_edges_per_bin": mean_edges_per_bin,
            "median_edges_per_bin": median_edges_per_bin,
        })

# =========================
# Write combined summaries
# =========================
if records_edge_counts:
    edge_df = pd.DataFrame(records_edge_counts).sort_values(["alpha","beta","bin"])
    edge_df.to_csv(OUT_DIR / "ALL_variants_edge_counts_edgewise.csv", index=False)
    print("\n-> ALL_variants_edge_counts_edgewise.csv")

if records_node_counts:
    node_df = pd.DataFrame(records_node_counts).sort_values(["alpha","beta","bin"])
    node_df.to_csv(OUT_DIR / "ALL_variants_node_counts_pairwise.csv", index=False)
    print("-> ALL_variants_node_counts_pairwise.csv")

sens_full = pd.DataFrame(records_sens_rows).drop_duplicates().sort_values(["alpha","beta"])
sens_full_path = OUT_DIR / "sensitivity_metrics.csv"
sens_full.to_csv(sens_full_path, index=False)
print("-> sensitivity_metrics.csv")

# =========================
# Plot ALL metrics (auto-tight z-range with ~15% padding)
# =========================
def _plot_metric_surface(metric_name: str, pretty: str):
    if sens_full.empty or metric_name not in sens_full.columns:
        print(f"[WARN] Skipping {metric_name}: no data.")
        return
    pivot = sens_full.pivot_table(index="alpha", columns="beta", values=metric_name)
    Z = pivot.to_numpy()
    if np.all(np.isnan(Z)):
        print(f"[WARN] Skipping {metric_name}: all NaN.")
        return
    A = pivot.index.to_numpy()
    B = pivot.columns.to_numpy()

    zmin = float(np.nanmin(Z)); zmax = float(np.nanmax(Z))
    pad  = max((zmax - zmin) * 0.15, 1e-6)
    zrange = [zmin - pad, zmax + pad]

    if _HAS_PLOTLY:
        fig = go.Figure(data=[go.Surface(z=Z, x=A, y=B)])
        fig.update_layout(
            title=f"Sensitivity: {pretty} vs α, β",
            scene=dict(
                xaxis_title="alpha",
                yaxis_title="beta",
                zaxis=dict(title=metric_name, range=zrange)
            ),
            autosize=True
        )
        html_path = OUT_DIR / f"sensitivity_{metric_name}_3D.html"
        fig.write_html(html_path)
        print(f"-> {html_path.name}")
    else:
        fig, ax = plt.subplots(figsize=(7,5))
        im = ax.imshow(
            Z, aspect="auto", origin="lower",
            extent=[B.min(), B.max(), A.min(), A.max()],
            vmin=zrange[0], vmax=zrange[1]
        )
        ax.set_xlabel("beta"); ax.set_ylabel("alpha")
        ax.set_title(pretty)
        cbar = plt.colorbar(im, ax=ax); cbar.set_label(metric_name)
        png_path = OUT_DIR / f"sensitivity_{metric_name}_heatmap.png"
        fig.tight_layout(); fig.savefig(png_path); plt.close(fig)
        print(f"-> {png_path.name}")

_metrics_to_plot = [
    ("coverage_rate", "coverage_rate"),
    ("mean_nodes_per_bin", "mean nodes per bin (pair-wise)"),
    ("effective_bins", "effective number of bins (pairs)"),
    ("separation_score", "adjacent-bin separation (pairs)"),
    ("median_nodes_per_bin", "median nodes per bin (pair-wise)"),
    ("trimmed_mean_nodes_per_bin", "trimmed-mean nodes per bin (20%) (pair-wise)"),
    ("node_evenness", "Pielou node evenness (dominant bin)"),
    ("node_gini", "node Gini (dominant bin)"),
    ("edge_coverage_rate", "edge coverage rate (edge-wise)"),
    ("mean_edges_per_bin", "mean edges per bin (edge-wise)"),
    ("median_edges_per_bin", "median edges per bin (edge-wise)"),
]

for m, title in _metrics_to_plot:
    _plot_metric_surface(m, title)

print("\n=== DONE ===")
print("Binning: tail-aware percentiles (0–50–75–85–90–95–97–98–99–99.5–100%), labels 1..10 (low→high).")
print("Sensitivity uses PAIR-WISE stats to avoid single busy pairs dominating.")
print("All metrics plotted with auto-tight z-ranges so small differences are visible.")
print("Ricci recomputation is skipped automatically when per-(α,β) CSV already exists.")
