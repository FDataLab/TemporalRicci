# TRicci: Temporal Ricci-Based Edge Sparsification for Dynamic Graph Learning

This repository contains the implementation of **TRicci**, a Temporal Forman–Ricci curvature-based edge sparsification framework for dynamic graph learning.

TRicci assigns an importance score to each directed, weighted, temporal edge by combining structural support, temporal proximity, and local interaction competition. The goal is to construct compact temporal graph representations that preserve downstream predictive performance while substantially reducing graph size and runtime.

## Overview

The framework processes temporal graphs as sequences of snapshots, computes TRicci scores for edges within each snapshot, ranks edges based on their curvature values, and retains the most informative curvature-ranked edges for downstream prediction.

The method is evaluated on graph-level temporal prediction tasks, including:

* Network activity growth prediction
* Network participation expansion prediction
* Influential node turnover prediction

Experiments are conducted on blockchain transaction networks and TGBL benchmark datasets.

## Key Features

* Temporal Forman–Ricci curvature computation for directed weighted temporal graphs
* Snapshot-level edge scoring and ranking
* Curvature-based edge sparsification
* Tau sensitivity analysis
* RNN-based temporal prediction using LSTM/GRU models
* ROC-AUC and runtime evaluation against baseline sparsification methods
