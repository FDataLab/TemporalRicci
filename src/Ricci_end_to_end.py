import pandas as pd
import temporalRicci
import vanillaRicci
import os
import matplotlib.pyplot as plt

def visualize_ricci_distribution(ricci_values, ricci_file_name, range):
    plt.figure(figsize=(6, 4))
    plt.hist(
        ricci_values,
        bins=50,
        color='lightblue',
        edgecolor='black',
        range=range
    )

    plt.xlabel('Ricci Curvature Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f"../results/{ricci_file_name}.png")

def create_ricci_data(dataset, graph):
    # Compute ricci values and receive dataframe of result
    formanRicci_df = temporalRicci.compute_forman_ricci(graph)
    formanRicci_df = formanRicci_df.rename(columns={'Result': 'Temporal Ricci value'})

    # vanillaRicci_df = vanillaRicci.compute_vanilla_ricci(graph)
    # vanillaRicci_df = vanillaRicci_df.rename(columns={'Result': 'Ricci value'})

    # # Joint 2 dataframes
    # ricci_df = pd.merge(formanRicci_df, on=['from', 'to', 'timestamp', 'value', 'blockNumber', 'tokenAddress', 'fileBlock', "original_timestamp"], how='inner')

    if not os.path.exists(f'../results/Ricci Values/{dataset}'):
        os.makedirs(f'../results/Ricci Values/{dataset}')
    print(f'../results/Ricci Values/{dataset}/{dataset}_Ricci_Values.csv')
    formanRicci_df.to_csv(f'../results/Ricci Values/{dataset}/{dataset}_Ricci_Values.csv', index=False)

    # visualize_ricci_distribution(ricci_df["Ricci value"], f"vanilla{dataset}Curvature_normalized", (-5000, 5000))
    visualize_ricci_distribution(formanRicci_df["Temporal Ricci value"], f"TFR{dataset}Curvature_normalized", (-5000, 5000))
