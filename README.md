# TRicci: Temporal Ricci-Based Edge Sparsification for Dynamic Graph Learning

This repository contains the implementation of **TRicci**, a Temporal Forman–Ricci curvature-based edge sparsification framework for dynamic graph learning.

TRicci assigns an importance score to each directed, weighted, temporal edge by combining structural support, temporal proximity, and local interaction competition. The goal is to construct compact temporal graph representations that preserve downstream predictive performance while substantially reducing graph size and runtime.

## Overview

The framework processes temporal graphs as sequences of snapshots, computes TRicci scores for edges within each snapshot, ranks edges based on their curvature values, and retains the most informative curvature-ranked edges for downstream prediction.

The method is evaluated on graph-level temporal prediction tasks, including:

- Network activity growth prediction
- Network participation expansion prediction
- Influential node turnover prediction

Experiments are conducted on blockchain transaction networks and TGBL benchmark datasets.

## Key Features

- Temporal Forman–Ricci curvature computation for directed, weighted temporal graphs
- Snapshot-level edge scoring and ranking
- Curvature-based edge sparsification
- Tau sensitivity analysis
- RNN-based temporal prediction using LSTM/GRU models
- ROC-AUC and runtime evaluation against baseline sparsification methods

## Analysis of Curvature Scores in a Local Neighborhood

To examine the behavior of curvature-based edge scoring in a local temporal neighborhood, the figure below presents a directed transaction subgraph with assigned edge weights and relative temporal positions. The highlighted edge $e_9$ is the target interaction selected for curvature analysis.

Edge color intensity indicates temporal proximity relative to the target edge: darker edges represent interactions occurring closer in time to $e_9$, while lighter edges represent more temporally distant interactions. The temporal encoding is therefore defined relative to the inspected edge rather than as a globally ordered timestamp axis.

![Local temporal neighborhood used for curvature analysis](figures/toy_graph.pdf)

*Local temporal neighborhood used to analyze the curvature of the highlighted target edge $e_9$.*

The following table reports both the proposed TRicci scores and the standard weighted Forman–Ricci curvature (FRC) scores for the edges in the local neighborhood.

| Edge | Date | Weight ($w_e$) | FRC | TRicci |
|:---:|:---:|---:|---:|---:|
| $e_6$ | Jan. 1 | 0.144 | -1.004 | 1.276 |
| $e_7$ | Jan. 3 | 0.265 | -2.200 | 0.914 |
| $e_8$ | Jan. 4 | 0.229 | -1.839 | 0.714 |
| **$e_9$** | **Jan. 5** | **0.225** | **-5.331** | **0.496** |
| $e_{10}$ | Jan. 6 | 0.238 | -3.610 | 0.828 |
| $e_{11}$ | Jan. 7 | 0.292 | -4.373 | 0.919 |
| $e_{12}$ | Jan. 8 | 0.281 | -3.894 | 1.280 |
| $e_{13}$ | Jan. 9 | 0.252 | -3.565 | 1.252 |

In this example, the target edge $e_9$ receives the lowest score under both formulations. This indicates that it is weakly supported relative to the surrounding interactions and is therefore a natural candidate for removal.

However, this agreement should not be interpreted as a general property of the two methods. Standard weighted FRC evaluates only weighted local topology and does not explicitly model temporal proximity or directed temporal competition. In contrast, TRicci incorporates relative temporal distance and competition among outgoing neighboring edges into the scoring process. Consequently, the two formulations may rank edges differently in more complex temporal settings, where strong static connectivity does not necessarily imply strong temporal relevance.
