import math
import pandas as pd
import networkx as nx
import time
from dateutil import parser
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

def vanillaForman(graph, edge):
    w_e = graph[edge[0]][edge[1]]['value']
    w_v1 = sum(graph[u][v]['value'] for u, v in graph.out_edges(edge[0]))
    w_v2 = sum(graph[u][v]['value'] for u, v in graph.out_edges(edge[1]))

    sum_term = 0

    for neighbor in graph.successors(edge[0]):
        if neighbor != edge[1]:  # Exclude the current edge
            w_e_v1 = graph[edge[0]][neighbor]['value']
            term = w_v1 / math.sqrt(w_e * w_e_v1)
            # print(f"Outgoing term to {neighbor}: {term}")
            sum_term += term

    for neighbor in graph.successors(edge[1]):
        if neighbor != edge[0]:  # Exclude the current edge
            w_e_v2 = graph[edge[1]][neighbor]['value']
            term = w_v2 / math.sqrt(w_e * w_e_v2)
            # print(f"Outgoing term from {neighbor}: {term}")
            sum_term += term

    F_e = w_e * ((w_v1 / w_e) + (w_v2 / w_e) - sum_term)
    return round(F_e, 2)

def compute_vanilla_ricci(graph):
    results = []
    for edge in tqdm(graph.edges(), desc="Vanilla: Processing edges"):
        result = vanillaForman(graph, edge)
        u = edge[0]
        v = edge[1]
        results.append((u, v, graph[u][v]['timestamp'], graph[u][v]['value'], result, 
                    graph[u][v]['blockNumber'] , graph[u][v]['tokenAddress'] , graph[u][v]['fileBlock'], graph[u][v]['original_timestamp']))

    results_df = pd.DataFrame(results, columns=['from', 'to', 'timestamp', 'value', 'Result', 'blockNumber', 'tokenAddress', 'fileBlock', 'original_timestamp'])
    
    return results_df