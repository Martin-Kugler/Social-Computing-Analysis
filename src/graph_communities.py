import networkx as nx
from pathlib import Path
import itertools
from infomap import Infomap
from graph_map_visualization import visualize_map_graph

def graph_communities_report(input_filename, output_filename, gn_divisions=20, random_seed=42, graph_name="graph"):
    """
    Generates a text report with community detection metrics (Louvain, Girvan-Newman and InfoMap)
    for any given graph and saves it to the 'results' directory, and saves the corresponding 
    visualizations to the 'results' directory.
    """
    
    # Extract the base directory route of the project adn establish the respective paths from data and results:
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_path = BASE_DIR / "data" / "processed" / input_filename
    results_path = BASE_DIR / "results"
    report_path = results_path / output_filename
    
    # Load the graph:
    print(f"Loading graph from {data_path}...")
    graph = nx.read_gexf(data_path)
    
    # Report lines: 
    report_lines = []
    report_lines.append(f"Metrics Report for: {input_filename}\n")
    report_lines.append("-"*50 + "\n\n")
    
    

    # 1) The Algorithm of Louvain:
    report_lines.append("### 1) The Algorithm of Louvain:\n")
    
    # 1.1) Calculate the communities detected by Louvain and print its length: 
    coms_louvain = list(nx.community.louvain_communities(graph, seed=random_seed, weight='weight'))
    report_lines.append(f"{len(coms_louvain)} Louvain communities detected\n")
    
    # 1.2) Calculate the modularity of this partition (to see how decent it is): 
    louvain_mod = nx.community.modularity(graph, coms_louvain, weight='weight')
    report_lines.append(f"Modularity = {louvain_mod}\n\n") # Decent modularity (bigger than 0.3)

    # 1.4) Visualize the communities-graph with a map-format, using
    # the respective "visualize_map_graph" in src: 

    # Create the respective dictionary for the communities calculated:
    dict_louvain = {i: list(c) for i, c in enumerate(coms_louvain)}
    print("Generating Louvain visualization...")
    
    fig_louvain = visualize_map_graph(
        input_filename=input_filename,
        threshold=500000, 
        title=f"{graph_name} Communities (Louvain)",
        communities=dict_louvain, 
        min_size=40
    )
    if fig_louvain is not None:
        out_plot_louvain = results_path / f"{input_filename.replace('.gexf', '')}_louvain.html"
        fig_louvain.write_html(str(out_plot_louvain))
        print(f"Louvain plot saved at {out_plot_louvain}")


    # 2) Algorithm of Girvan-Newman:
    report_lines.append("### 2) Algorithm of Girvan-Newman:\n")
    
    # 2.1) Calculate the communities detected by Grivan-Newman and print its length and modularity: 
    # Create the attribute "distance":
    for u, v, d in graph.edges(data=True):
        # We add a minuscule value (1e-6) to avoid divisions by zero in case any weight is 0:
        graph[u][v]['distance'] = 1 / (d['weight'] + 1e-6)

    # We define a function that tells GN how to calculate the weight of the edges:
    def heaviest_edge_betweenness(g):
        centrality = nx.edge_betweenness_centrality(g, weight='distance')
        return max(centrality, key=centrality.get)

    gen_gn = nx.community.girvan_newman(graph, most_valuable_edge=heaviest_edge_betweenness)

    # Evaluate the first n divisions and select the one with the higher modularity: 
    gn_mod = -1.0
    best_partition_gn = None

    for partition in itertools.islice(gen_gn, gn_divisions):
        mod = nx.community.modularity(graph, partition, weight='weight')
        if mod > gn_mod:
            gn_mod = mod
            best_partition_gn = partition

    if best_partition_gn:
        report_lines.append(f"{len(best_partition_gn)} Girvan-Newman communities detected\n")
        report_lines.append(f"Modularity = {gn_mod}\n\n") # Dreadful modularity.

        # 2.2) Visualize the communities-graph with a map-format, using
        # the respective "visualize_map_graph" in src: 

        # Create the respective dictionary for the communities calculated:
        dict_gn = {i: list(c) for i, c in enumerate(best_partition_gn)}
        print("Generating Girvan-Newman visualization...")
        
        fig_gn = visualize_map_graph(
            input_filename=input_filename,
            threshold=500000,
            title=f"{graph_name} Communities (Girvan-Newman)",
            communities=dict_gn, 
            min_size=40
        )
        if fig_gn is not None:
            out_plot_gn = results_path / f"{input_filename.replace('.gexf', '')}_gn.html"
            fig_gn.write_html(str(out_plot_gn))
            print(f"Girvan-Newman plot saved at {out_plot_gn}")
    


    # 3) InfoMap:
    report_lines.append("### 3) InfoMap Algorithm:\n")
    
    # 3.1) Calculate the communities detected by InfoMap and print its length: 
    # InfoMap expects the nodes as numbers: 
    node_names = list(graph.nodes())
    map_id = {name: i for i, name in enumerate(node_names)}

    # Initialize the InfoMap object: 
    im = Infomap(silent=True, directed=True, two_level=True)

    # Add the respective edges:
    for u, v, d in graph.edges(data=True):
        im.add_link(map_id[u], map_id[v], d['weight'])

    # Execute the algorithm: 
    im.run()

    # Extract the communities as a dictionary iterating the tree given by InfoMap: 
    dict_coms_im = {}
    for node in im.tree:
        if node.is_leaf:
            mod_id = node.module_id

            # Each country (iso) must be included in its community:
            if mod_id not in dict_coms_im:
                dict_coms_im[mod_id] = set()

            iso = node_names[node.node_id] 
            dict_coms_im[node.module_id].add(iso)

    coms_im = list(dict_coms_im.values())

    report_lines.append(f"{im.num_top_modules} InfoMap communities detected\n")

    # 3.2) Calculate the modularity of this partition: 
    im_mod = nx.community.modularity(graph, coms_im, weight='weight')
    report_lines.append(f"Modularity = {im_mod}\n\n") # Decent modularity (bigger than 0.3)

    # 3.3) Visualize the communities-graph with a map-format, using
    # the respective "visualize_map_graph" in src: 

    # Create the respective dictionary for the communities calculated:
    print("Generating InfoMap visualization...")
    fig_im = visualize_map_graph(
        input_filename=input_filename,
        threshold=500000,
        title=f"{graph_name} Communities (InfoMap)",
        communities=dict_coms_im, 
        min_size=40
    )
    if fig_im is not None:
        out_plot_im = results_path / f"{input_filename.replace('.gexf', '')}_infomap.html"
        fig_im.write_html(str(out_plot_im))
        print(f"InfoMap plot saved at {out_plot_im}")



    # 4) Save the report in results as a .txt:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
        
    print(f"Report generated successfully. Saved at: {report_path}")


if __name__ == '__main__':
    # Testing:
    graph_communities_report(
        input_filename="world_trade_network_petrol_2024.gexf", 
        output_filename="communities_metrics_report_2024.txt",
        gn_divisions=20, 
        graph_name='Petrol2024'
    )