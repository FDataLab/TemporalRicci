# Temporal-Ricci-Curvature4

This repository includes tools to:
- **Compute and bin temporal Forman–Ricci curvature** for blockchain transaction networks.  
  The curvature values are calculated per edge over time and then partitioned into multiple bins to analyze how different curvature ranges contribute to network dynamics.
- **Generate daily network snapshots** and extract topological features*. These snapshots capture evolving structural patterns that can be mapped into feature sequences for machine learning.
- **Label and prepare datasets** for three predictive tasks that capture the structural evolution of dynamic graphs.

---

## Prediction Tasks

Each dataset is converted into time-windowed graph sequences and labeled for one of the following tasks:

### **Task 1 — Network Growth Prediction**
Determines whether the number of edges in the network will increase in the following time window.  
This task reflects the short-term expansion or contraction of the transaction network.

### **Task 2 — Influential Node Count Prediction**
Estimates whether the count of top-1% high-volume (most active) nodes will increase in the next time window.  
It focuses on identifying shifts in activity concentration and the emergence of influential participants.

### **Task 3 — Connected Components Prediction**
Predicts whether the number of connected components in the graph will increase, indicating network fragmentation or reduced connectivity.

---

## Pipeline Summary

1. **Input**  
   Precomputed curvature-based CSV files are stored under:
   RicciResults/ricci_values/<DATASET>/
   The dataset name should follow the pattern:
    <DATASET>_TFR_a<ALPHA>_b<BETA>.csv
    <DATASET>_TFR_a<ALPHA>_b<BETA>_bin1.csv
    ...
    <DATASET>_TFR_a<ALPHA>_b<BETA>_bin10.csv
   
2. **Sequence Generation**  
The script [`network_parser.py`](src/GraphPulse/GraphPulse/analyzer/network_parser.py) processes each dataset and creates **time-windowed network sequences** in two formats:  
- **TDA-based sequences** — topological descriptors extracted using *KeplerMapper*.
- **Raw sequences** — basic structural statistics such as node count, edge count, and average degree per window.  

3. **Labeling**  
Each generated sequence is labeled automatically according to the three predictive task definitions:  
- **Task 1:** Network Growth  
- **Task 2:** Influential Node Count  
- **Task 3:** Connected Components  

4. **Output**  
- **Feature sequences** (`seq_tda.txt` and `seq_raw.txt`) are saved under:  
  ```
  GraphPulseResults/
    ├── Sequence_task1/
    ├── Sequence_task2/
    └── Sequence_task3/
  ```  
  Each task folder contains subdirectories for all processed datasets.  
- **Runtime and status logs** are saved to:  
  ```
  GraphPulseResults/<DATASET>_run_times.csv
  ```      
