import os
import glob
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def fmt_val(x):
    s = f"{float(x):.6g}"
    return s.replace('.', 'p').replace('-', 'm')

def load_matrices(outdir):
    z_bins_path  = os.path.join(outdir, "Z_bins_nonempty.csv")
    z_nodes_path = os.path.join(outdir, "Z_mean_nodes.csv")
    Z_bins  = pd.read_csv(z_bins_path, index_col=0)
    Z_nodes = pd.read_csv(z_nodes_path, index_col=0)

    # axes
    alphas = np.array(Z_bins.index.astype(float).tolist())
    betas  = np.array(Z_bins.columns.astype(float).tolist())
    Zb = Z_bins.values
    Zn = Z_nodes.values
    return alphas, betas, Zb, Zn

def plot_interactive_surfaces(outdir, alphas, betas, Zb, Zn):
    if not PLOTLY_OK:
        print("[WARN] Plotly not installed; skipping interactive HTML surfaces.")
        return

    A, B = np.meshgrid(betas, alphas)  # just to match shapes if needed

    # 3D Surface: # non-empty bins (X=alpha, Y=beta)
    fig1 = go.Figure(data=[go.Surface(
        x=alphas, y=betas, z=Zb.T, contours={"z": {"show": True, "usecolormap": True}}
    )])
    fig1.update_layout(
        title="# non-empty bins vs alpha/beta",
        scene=dict(
            xaxis_title="alpha",
            yaxis_title="beta",
            zaxis_title="# non-empty bins",
        ),
        autosize=True,
    )
    fig1.write_html(os.path.join(outdir, "surface_bins_interactive.html"))

    # 3D Surface: mean nodes/bin (X=alpha, Y=beta)
    fig2 = go.Figure(data=[go.Surface(
        x=alphas, y=betas, z=Zn.T, contours={"z": {"show": True, "usecolormap": True}}
    )])
    fig2.update_layout(
        title="mean #nodes/bin vs alpha/beta",
        scene=dict(
            xaxis_title="alpha",
            yaxis_title="beta",
            zaxis_title="mean #nodes/bin",
        ),
        autosize=True,
    )
    fig2.write_html(os.path.join(outdir, "surface_nodes_interactive.html"))

def plot_heatmaps(outdir, alphas, betas, Zb, Zn):
    plt.figure(figsize=(8,6))
    plt.imshow(Zb.T, origin="lower", aspect="auto",
               extent=[alphas.min(), alphas.max(), betas.min(), betas.max()])
    plt.xlabel("alpha"); plt.ylabel("beta"); plt.title("# non-empty bins (heatmap)")
    cbar = plt.colorbar(); cbar.set_label("# non-empty bins")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "heatmap_bins.png"), dpi=180)

    # Heatmap: mean nodes/bin
    plt.figure(figsize=(8,6))
    plt.imshow(Zn.T, origin="lower", aspect="auto",
               extent=[alphas.min(), alphas.max(), betas.min(), betas.max()])
    plt.xlabel("alpha"); plt.ylabel("beta"); plt.title("mean #nodes/bin (heatmap)")
    cbar = plt.colorbar(); cbar.set_label("mean #nodes/bin")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "heatmap_nodes.png"), dpi=180)

def entropy(p):
    # p: 1D array of counts (edges or nodes per bin). Returns Shannon entropy in nats.
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if s <= 0:
        return 0.0
    q = p / s
    q = q[q > 0]
    return -np.sum(q * np.log(q))

def scan_spread(outdir, nbins_guess=10):
    """
    Walk ../RicciResults/Sensitivity/<dataset>/<mode>/grid/alpha_*/beta_*/bin_stats.csv
    and compute "spread" for each (alpha,beta) as entropy over per-bin edge counts.
    Returns a DataFrame with metrics and best rows.
    """
    grid_root = os.path.join(outdir, "grid")
    entries = []
    for alpha_dir in sorted(glob.glob(os.path.join(grid_root, "alpha_*"))):
        alpha_tok = os.path.basename(alpha_dir).split("alpha_")[-1]
        try: alpha_val = float(alpha_tok.replace('p','.')
                                        .replace('m','-'))
        except: continue
        for beta_dir in sorted(glob.glob(os.path.join(alpha_dir, "beta_*"))):
            beta_tok = os.path.basename(beta_dir).split("beta_")[-1]
            try: beta_val = float(beta_tok.replace('p','.')
                                        .replace('m','-'))
            except: continue

            stats_path = os.path.join(beta_dir, "bin_stats.csv")
            if not os.path.exists(stats_path):
                continue

            df = pd.read_csv(stats_path)
            if "bin" in df.columns:
                df = df.sort_values("bin")

            edges_per_bin = df["edges"].to_numpy() if "edges" in df.columns else np.zeros(nbins_guess)
            nodes_per_bin = df["nodes"].to_numpy() if "nodes" in df.columns else np.zeros(nbins_guess)

            nonempty_bins = int((edges_per_bin > 0).sum())
            mean_nodes_nonempty = float(nodes_per_bin[edges_per_bin > 0].mean()) if (edges_per_bin > 0).any() else 0.0

            # Entropy over edges distribution across bins (higher = more spread)
            H_edges = float(entropy(edges_per_bin))
            # Entropy over nodes distribution, too (optional)
            H_nodes = float(entropy(nodes_per_bin))

            entries.append({
                "alpha": alpha_val,
                "beta": beta_val,
                "nonempty_bins": nonempty_bins,
                "mean_nodes_nonempty_bin": mean_nodes_nonempty,
                "entropy_edges": H_edges,
                "entropy_nodes": H_nodes,
                "total_edges": int(edges_per_bin.sum()),
                "total_nodes": int(nodes_per_bin.sum())
            })

    if not entries:
        return None, None

    summary = pd.DataFrame(entries)

    best_nonempty = summary.sort_values(["nonempty_bins","entropy_edges","mean_nodes_nonempty_bin"], ascending=[False, False, False]).head(5)
    best_entropy  = summary.sort_values(["entropy_edges","nonempty_bins"], ascending=[False, False]).head(5)
    best_nodes    = summary.sort_values(["mean_nodes_nonempty_bin","nonempty_bins"], ascending=[False, False]).head(5)

    best_dir = os.path.join(outdir, "best")
    ensure_dir(best_dir)
    summary.to_csv(os.path.join(best_dir, "summary_all_metrics.csv"), index=False)
    best_nonempty.to_csv(os.path.join(best_dir, "top_nonempty_bins.csv"), index=False)
    best_entropy.to_csv(os.path.join(best_dir, "top_entropy_edges.csv"), index=False)
    best_nodes.to_csv(os.path.join(best_dir, "top_mean_nodes.csv"), index=False)

    return summary, {
        "top_nonempty_bins": best_nonempty,
        "top_entropy_edges": best_entropy,
        "top_mean_nodes": best_nodes
    }

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g., DERC (without .csv)")
    ap.add_argument("--mode", choices=["additive","log"], default="additive")
    args = ap.parse_args()

    outdir = os.path.join("..", "RicciResults", "Sensitivity", args.dataset, args.mode)
    if not os.path.exists(outdir):
        raise SystemExit(f"Results folder not found: {outdir}")

    print(f"[INFO] Loading matrices from {outdir}")
    alphas, betas, Zb, Zn = load_matrices(outdir)

    print("[INFO] Making interactive 3D surfaces (HTML)")
    plot_interactive_surfaces(outdir, alphas, betas, Zb, Zn)

    print("[INFO] Making heatmaps (PNG)")
    plot_heatmaps(outdir, alphas, betas, Zb, Zn)

    print("[INFO] Scanning grid for spread metrics and best (alpha,beta)")
    summary, best = scan_spread(outdir)
    if summary is None:
        print("[WARN] No bin_stats.csv files found under grid/. Did you run the sweep with saving bins?")
    else:
        print("[INFO] Wrote best/*.csv with top candidates.")
        print(best["top_entropy_edges"].head(3).to_string(index=False))
        print(best["top_nonempty_bins"].head(3).to_string(index=False))
        print(best["top_mean_nodes"].head(3).to_string(index=False))

if __name__ == "__main__":
    main()
