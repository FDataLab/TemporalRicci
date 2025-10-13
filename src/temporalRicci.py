# temporalRicci.py
import math
import pandas as pd
from tqdm import tqdm
import networkx as nx
import numpy as np
from normalization import normalize_deltaT

# deterministic RNG for sampling
_RNG = np.random.default_rng(0)

def _edge_time(G, e):  # e = (u,v,k)
    return G[e[0]][e[1]][e[2]]['timestamp']

def _edge_val(G, e):   # e = (u,v,k)
    return G[e[0]][e[1]][e[2]]['value']

def calculate_edge_diff_gap(G, e1, e2):
    """
    Δt for two *edges with keys* e1=(u,v,k), e2=(u,v,k):
    - if timestamps differ: |t2 - t1|
    - else: fractional intra-timestamp spacing based on index distance
    """
    t1 = _edge_time(G, e1)
    t2 = _edge_time(G, e2)
    if t1 != t2:
        return abs(t2 - t1)
    # same timestamp: small but nonzero distance using order index
    tx = [(u, v, k) for u, v, k in G.edges(keys=True) if _edge_time(G, (u, v, k)) == t1]
    tx_sorted = sorted(tx)  # deterministic
    idx = {e: i for i, e in enumerate(tx_sorted)}
    n = len(tx_sorted)
    return abs(idx[e1] - idx[e2]) / (n + 1.0)

def _compute_node_weight(G, n, use_in_and_out=True):
    tot = 0.0
    if use_in_and_out:
        for u, v, k in G.in_edges(n, keys=True):
            tot += _edge_val(G, (u, v, k))
        for u, v, k in G.out_edges(n, keys=True):
            tot += _edge_val(G, (u, v, k))
    else:
        for u, v, k in G.out_edges(n, keys=True):
            tot += _edge_val(G, (u, v, k))
    return tot

def compute_w_values(G, use_in_and_out=True):
    return {n: _compute_node_weight(G, n, use_in_and_out) for n in G.nodes()}

def _compute_sum_term(G, e, node, other, w_e, w_values, alpha, beta, min_gap_norm=1e-9,
                      max_samples=100):
    """
    Sum over out-neighbors of `node` (excluding `other`) with temporal kernel K(Δt)=exp(-alpha * Δt / beta).
    Uses per-edge weights and keys.
    """
    nbrs = [nbr for nbr in G.successors(node) if nbr != other]
    if not nbrs:
        return 0.0
    if len(nbrs) > max_samples:
        nbrs = _RNG.choice(nbrs, max_samples, replace=False)

    sum_term = 0.0
    for nbr in nbrs:
        # iterate all parallel edges to neighbor
        for k2 in G[node][nbr].keys():
            e2 = (node, nbr, k2)
            w_en = _edge_val(G, e2)

            dt_raw = calculate_edge_diff_gap(G, e2, e)
            # normalize Δt in your preferred way (keep your function)
            dt_norm = normalize_deltaT(dt_raw, 10)
            dt_norm = max(dt_norm, min_gap_norm)

            # stable monotone kernel; beta>0 acts as time scale
            K = math.exp(-alpha * (dt_norm / max(beta, 1e-12)))

            # multiply by K; avoid divide-by-zero in sqrt
            denom = math.sqrt(max(w_e, 1e-12) * max(w_en, 1e-12))
            sum_term += (w_values[node] * K) / denom

    return float(sum_term)

def temporalForman(G, e, w_values, alpha, beta):
    """
    Temporal Forman-like curvature for edge e=(u,v,k), using:
    TF_e = w_e * ((w_u / w_e) + (w_v / w_e) - (S_u + S_v))
    where S_u,S_v are temporally-weighted neighbor sums.
    """
    u, v, k = e
    w_e = _edge_val(G, e)
    w_u = w_values.get(u, 0.0)
    w_v = w_values.get(v, 0.0)

    S_u = _compute_sum_term(G, e, u, v, w_e, w_values, alpha, beta)
    S_v = _compute_sum_term(G, e, v, u, w_e, w_values, alpha, beta)

    TF_e = w_e * ((w_u / max(w_e, 1e-12)) + (w_v / max(w_e, 1e-12)) - (S_u + S_v))
    return TF_e

def compute_forman_ricci(G, alpha=0.1, beta=1.0):
    """
    Compute temporal curvature for all edges of a (Multi)DiGraph.
    Always iterate with keys to be robust to multi-edges.
    """
    w_values = compute_w_values(G, use_in_and_out=True)
    rows = []
    for u, v, k, d in tqdm(G.edges(keys=True, data=True), desc=f"TFR a={alpha} b={beta}"):
        tf = temporalForman(G, (u, v, k), w_values, alpha, beta)
        rows.append((u, v, d.get('timestamp', None), d.get('value', None), tf))
    return pd.DataFrame(rows, columns=['from', 'to', 'timestamp', 'value', 'Temporal Ricci value'])
