import networkx as nx
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import json

def graph_metrics_report(graph_path, output_dir="results", graph_name="graph"):
    """
    Reads a directed weighted graph and generates a complete metrics report,
    exporting dataframes, global metrics, and plots to the specified output directory.
    """

    # Create output directory if it doesn't exist:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading graph from {graph_path}...")
    graph = nx.read_gexf(graph_path)
    

    # 1) DEGREE ANALYSIS: 
    print("Calculating degree metrics...")

    # 1.1) Nodes degree (considering the weight and the in and out degree):
    # In order to visualize these metrics better, we shall display them as a Pandas DataFrame: 
    metrics = []
    for node in graph.nodes():
        metrics.append({
            'Country': node,
            # Diplomacy (number of associates -> Without considering weights): 
            'Num_Exports': graph.out_degree(node),
            'Num_Imports': graph.in_degree(node),
            # Economy (volume of trades -> considering weights):
            'Amount_Exports': graph.out_degree(node, weight='weight'),
            'Amount_Imports': graph.in_degree(node, weight='weight')
        })

    df_metrics = pd.DataFrame(metrics).sort_values(by='Amount_Exports', ascending=False)
    
    # 1.2) Average degrees:
    average_degrees = df_metrics[['Num_Exports', 'Num_Imports', 'Amount_Exports', 'Amount_Imports']].mean()
    
    # 1.3) Degree distribution plot:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(2, 2, figsize=(18, 12))

    # Declare the maximum of number of exports and imports in order to specify a top
    # when assigning the bins in the respective histograms:
    max_exports = int(df_metrics['Num_Exports'].max())
    max_imports = int(df_metrics['Num_Imports'].max())

    # Number of exports:
    sns.histplot(data=df_metrics, x='Num_Exports', bins=range(0, max_exports + 2),
                 discrete=True, kde=True, color='#d1495b', alpha=0.8, ax=ax[0, 0])
    ax[0, 0].set_title('Out-degree (number of export partners)')
    ax[0, 0].set_xlabel('Num_Exports')
    ax[0, 0].set_ylabel('Number of countries')

    # Number of imports:
    sns.histplot(data=df_metrics, x='Num_Imports', bins=range(0, max_imports + 2),
                 discrete=True, kde=True, color='#0077b6', alpha=0.8, ax=ax[1, 0])
    ax[1, 0].set_title('In-degree (number of import partners)')
    ax[1, 0].set_xlabel('Num_Imports')
    ax[1, 0].set_ylabel('Number of countries')

    # Amount of exports:
    sns.histplot(data=df_metrics, x='Amount_Exports', bins=30,
                 kde=True, color='#e76f51', alpha=0.75, ax=ax[0, 1])
    ax[0, 1].set_title('Weighted out-degree (export volume)')
    ax[0, 1].set_xlabel('Amount_Exports')
    ax[0, 1].set_ylabel('Number of Countries')
    ax[0, 1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Amount of imports:
    sns.histplot(data=df_metrics, x='Amount_Imports', bins=30,
                 kde=True, color='#219ebc', alpha=0.75, ax=ax[1, 1])
    ax[1, 1].set_title('Weighted in-degree (import volume)')
    ax[1, 1].set_xlabel('Amount_Imports')
    ax[1, 1].set_ylabel('Number of Countries')
    ax[1, 1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    for axis in ax.flat:
        axis.grid(alpha=0.25)

    fig.suptitle(f'Degree Distributions of {graph_name}', fontsize=18, y=1.02)
    plt.tight_layout()
    
    # Save plot:
    degree_plot_path = out_path / f"{graph_name}_degree_distributions.png"
    plt.savefig(degree_plot_path, bbox_inches='tight')
    plt.close()


    # 2) BETWEENNESS:
    print("Calculating betweenness centrality...")

    # Here we must consider something important - the betweenness centrality calculates the number 
    # of minimum paths in an specific node. The thing is that this minimum path is calculated according to
    # the weights, so that an edge which presents a huge flow of goods would be considered as a "path with huge cost", 
    # but we know that in economy it means the opposite. Thus, we will be using the inverse of the weight as the distance:
    for u, v, d in graph.edges(data=True):
        # We add a minuscule value (1e-6) to avoid divisions by zero in case any weight is 0:
        graph[u][v]['distance'] = 1 / (d.get('weight', 1) + 1e-6)

    # Calculate the betweennes centrality considering the "distance" as the weight:
    betweenness = nx.betweenness_centrality(graph, weight='distance', normalized=True)

    # Save it in the metrics DataFrame: 
    df_metrics['Betweenness'] = df_metrics['Country'].map(betweenness)


    # 3) MOTIFS:
    print("Calculating clustering coefficients...")

    # 3.1) Local clustering coefficient: 
    local_clustering = nx.clustering(graph)
    df_metrics['Local_Clustering'] = df_metrics['Country'].map(local_clustering)

    # 3.2) Global clustering coefficient (transitivity):
    transitivity = nx.transitivity(graph)


    # 4) RECIPROCIY: 
    print("Calculating reciprocity...")
    reciprocity = nx.reciprocity(graph)


    # 5) GEODETIC PATHS AND POSITION: 
    print("Calculating geodetic paths (this might take a moment)...")

    # For these next metrics we shall calculate the Giant Strongly Connected Component
    # in order to prevent any disconnection-type errors: 
    scc = max(nx.strongly_connected_components(graph), key=len)
    graph_scc = graph.subgraph(scc).copy()

    # 5.1) Diameter:
    diameter = nx.diameter(graph_scc, weight='distance')
    
    # 5.2) Eccentricity:
    eccentricity = nx.eccentricity(graph_scc, weight='distance')
    df_metrics['Eccentricity'] = df_metrics['Country'].map(eccentricity)

    # 5.3) Average geodesic path:
    avg_path = nx.average_shortest_path_length(graph_scc, weight='distance')

    # Distribution of geodetic path lengths:
    paths = dict(nx.all_pairs_dijkstra_path_length(graph_scc, weight='distance'))
    distances = []
    for source, targets in paths.items():
        for target, length in targets.items():
            if source != target:
                distances.append(length)

    plt.figure(figsize=(10, 6))
    sns.histplot(distances, kde=True, color='green', bins=30, edgecolor='black')
    plt.axvline(avg_path, color='red', linestyle='--', label=f'Average: {avg_path:.6f}')
    plt.axvline(diameter, color='yellow', linestyle='--', label=f'Diameter: {diameter:.6f}')

    plt.title(f'Distribution of geodetic path lengths ({graph_name})')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save geodetic paths plot:
    geodetic_plot_path = out_path / f"{graph_name}_geodetic_paths.png"
    plt.savefig(geodetic_plot_path, bbox_inches='tight')
    plt.close()

    
    # 6) COMPONENTS: 
    print("Calculating components...")

    # 6.1) Connected components analysis (for directed graphs):
    weak_components = list(nx.weakly_connected_components(graph))
    strong_components = list(nx.strongly_connected_components(graph))

    components_info = {
        "num_weak_components": len(weak_components),
        "num_strong_components": len(strong_components),
        "largest_weak_component_size": len(max(weak_components, key=len)),
        "largest_strong_component_size": len(max(strong_components, key=len)),
    }

    
    # 7) EXPORT DATA: 
    print("Exporting results...")

    # 7.1) Export node metrics into CSV:
    csv_path = out_path / f"{graph_name}_node_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)

    # Aggregate global metrics into a dictionary:
    global_metrics = {
        "transitivity": transitivity,
        "reciprocity": reciprocity,
        "diameter_scc": diameter,
        "average_geodesic_path_scc": avg_path,
        "average_degrees": average_degrees.to_dict(),
        "components": components_info
    }

    # 7.2) Export global metrics into JSON
    json_path = out_path / f"{graph_name}_global_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(global_metrics, f, indent=4)

    print(f"Report generation complete. Check the '{output_dir}' folder.")

if __name__ == "__main__":

    # Example usage:
    BASE_DIR = Path.cwd() 
    input_file = BASE_DIR / "data" / "processed" / "world_trade_network_petrol_2024.gexf"
    
    if input_file.exists():
        graph_metrics_report(
            graph_path=input_file, 
            output_dir="results", 
            graph_name="petrol_2024"
        )
    else:
        print(f"File not found: {input_file}. Please check the path.")

