import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def random_graphs_comparison_report(
	graph_path,
	output_dir="results",
	graph_name="graph",
	ws_rewire_prob=0.1,
	random_seed=42,
):
	"""
	Compares a real graph against random graph models (ER, BA, WS),
	and exports plots + summary metrics into the specified output directory.
	"""

	# Create output directory if it doesn't exist:
	out_path = Path(output_dir)
	out_path.mkdir(parents=True, exist_ok=True)

	print(f"Loading graph from {graph_path}...")
	graph = nx.read_gexf(graph_path)

	# 1) Graph preparation:
	# 1.1) Since the classic theoretical models of random graphs assume non-directed and without weights graphs,
	# we shall modify our graph according to it as well, considering only its Giant Connected Component (GCC):
	graph = nx.Graph(graph)
	gcc_nodes = max(nx.connected_components(graph), key=len)
	graph = graph.subgraph(gcc_nodes).copy()

	# 1.2) Extract the base metrics that will be used to compare our graph to the random ones:
	N = graph.number_of_nodes()
	E = graph.number_of_edges()
	k_avg = 2 * E / N  # Average degree.

	# 1.3) Calculate the key metrics of our graph:
	C = nx.average_clustering(graph)
	L = nx.average_shortest_path_length(graph)

	# 2) Erdös-Rényi model (ER):
	# 2.1) Calculate the probability p of connecting two nodes in the model according to the
	# base metrics of our graph and using the formula c = p * (n-1), in which c is the average degree:
	p_er = k_avg / (N - 1)

	# 2.2) Create the graph and select only its GCC:
	G_er = nx.erdos_renyi_graph(n=N, p=p_er, seed=random_seed)
	gcc_er = max(nx.connected_components(G_er), key=len)
	G_er = G_er.subgraph(gcc_er).copy()

	# 2.3) Calculate its key metrics:
	C_er = nx.average_clustering(G_er)
	L_er = nx.average_shortest_path_length(G_er)

	print("Erdős-Rényi Model:")
	print(f"Clustering ER: {C_er:.4f} | Clustering Real Graph: {C:.4f}")
	print(f"Path Length ER: {L_er:.4f} | Path Length Real Graph: {L:.4f}")

	# 3) Barabási-Albert model (BA):
	# 3.1) Calculate the average number of edges per nodes in our graph, using it to
	# initialize the algorithm with that number as m:
	M = graph.number_of_edges()
	M_INIT = int(round(M / N))
	M_INIT = max(M_INIT, 1)

	# 3.2) Create the graph:
	G_ba = nx.barabasi_albert_graph(N, M_INIT, seed=random_seed)

	# 3.3) Calculate its key metrics:
	C_ba = nx.average_clustering(G_ba)
	L_ba = nx.average_shortest_path_length(G_ba)

	print("Barabási-Albert Model:")
	print(f"Clustering BA: {C_ba:.4f} | Clustering Real Graph: {C:.4f}")
	print(f"Path Length BA: {L_ba:.4f} | Path Length Real Graph: {L:.4f}")

	# 4) Watts-Strogatz model (WS):
	# 4.1) Calculate the k (number of connections of each node with its neighbours)
	# just by converting our average degree (k_avg) in an integer:
	k_ws = int(round(k_avg))
	k_ws = max(k_ws, 2)
	if k_ws >= N:
		k_ws = N - 1 if (N - 1) % 2 == 0 else N - 2
	if k_ws % 2 != 0:
		k_ws -= 1

	# 4.2) Create the graph, establishing the probability of reconnecting (making a
	# shortcut) by 0.1 (the default for simulating the "Small World"):
	p_ws = ws_rewire_prob
	G_ws = nx.watts_strogatz_graph(n=N, k=k_ws, p=p_ws, seed=random_seed)

	# 4.3) Calculate its key metrics:
	if nx.is_connected(G_ws):
		C_ws = nx.average_clustering(G_ws)
		L_ws = nx.average_shortest_path_length(G_ws)
	else:
		gcc_ws = max(nx.connected_components(G_ws), key=len)
		G_ws = G_ws.subgraph(gcc_ws).copy()
		C_ws = nx.average_clustering(G_ws)
		L_ws = nx.average_shortest_path_length(G_ws)

	print("Watts-Strogatz Model:")
	print(f"Clustering WS: {C_ws:.4f} | Clustering Real Graph: {C:.4f}")
	print(f"Path Length WS: {L_ws:.4f} | Path Length Real Graph: {L:.4f}")

	# 5.1) Auxiliary functions to extract all metrics in list form:
	def get_degrees(G):
		return list(dict(G.degree()).values())

	def get_clusterings(G):
		clustering_values = nx.clustering(G)
		if isinstance(clustering_values, dict):
			return list(clustering_values.values())
		return [float(clustering_values)]

	def get_path_lengths(G):
		# Extracts all distances between all pairs of nodes (excluding the distance to itself):
		lengths = []
		for source, paths in nx.all_pairs_shortest_path_length(G):
			lengths.extend([l for target, l in paths.items() if source != target])
		return lengths

	# 5.2) Extract these metrics from the real graph:
	deg_real = get_degrees(graph)
	clus_real = get_clusterings(graph)
	path_real = get_path_lengths(graph)

	# 5.3) Pair the random models in a list to iterate over the columns:
	models = [
		("Erdős-Rényi Model", G_er),
		("Barabási-Albert Model", G_ba),
		("Watts-Strogatz Model", G_ws),
	]

	# 5.4) Create the matrix of subplots 3x3 (each row a distinct metric
	# and each column a distinct random graph, comparing them with the original one):
	fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 18))
	fig.suptitle(
		f"Distribution Analysis: {graph_name} vs Random Models",
		fontsize=18,
		fontweight="bold",
		y=0.96,
	)

	# 5.5) Define the loop which goes through each model and fill in its corresponding column:
	for col, (name_model, random_graph) in enumerate(models):

		# Extract the metrics from the respective model:
		deg_rand = get_degrees(random_graph)
		clus_rand = get_clusterings(random_graph)
		path_rand = get_path_lengths(random_graph)

		# ROW 0 - DEGREE DISTRIBUTION:
		axes[0, col].hist(deg_real, bins=20, alpha=0.5, density=True, label="Original", color="#1f77b4")
		axes[0, col].hist(deg_rand, bins=20, alpha=0.5, density=True, label="Model", color="#ff7f0e")
		axes[0, col].set_title(name_model, fontsize=14, pad=15)

		if col == 0:
			axes[0, col].set_ylabel("Probability", fontweight="bold")

		axes[0, col].set_xlabel("Degree (k)")
		axes[0, col].set_yscale("log")
		axes[0, col].legend()

		# ROW 1 - CLUSTERING DISTRIBUTION:
		axes[1, col].hist(clus_real, bins=20, alpha=0.5, density=True, label="Original", color="#1f77b4")
		axes[1, col].hist(clus_rand, bins=20, alpha=0.5, density=True, label="Model", color="#2ca02c")

		if col == 0:
			axes[1, col].set_ylabel("Normalized Frequency", fontweight="bold")

		axes[1, col].set_xlabel("Local Clustering Coefficient (C)")
		axes[1, col].legend()

		# ROW 2 - GEODETIC PATHS LENGTH DISTRIBUTION:
		bins_path = range(1, max(max(path_real), max(path_rand)) + 2)

		axes[2, col].hist(
			path_real,
			bins=bins_path,
			alpha=0.5,
			density=True,
			align="left",
			label="Original",
			color="#1f77b4",
		)
		axes[2, col].hist(
			path_rand,
			bins=bins_path,
			alpha=0.5,
			density=True,
			align="left",
			label="Model",
			color="#d62728",
		)

		if col == 0:
			axes[2, col].set_ylabel("Normalized Frequency", fontweight="bold")

		axes[2, col].set_xlabel("Geodetic Length (L)")
		axes[2, col].legend()

	# Adjust spaces and export the figure:
	plt.tight_layout(rect=[0, 0, 1, 0.94])
	figure_path = out_path / f"{graph_name}_random_models_distributions.png"
	plt.savefig(figure_path, bbox_inches="tight")
	plt.close(fig)

	# 6) Export summary report:
	summary = {
		"graph_name": graph_name,
		"base_graph": {
			"nodes": N,
			"edges": E,
			"average_degree": k_avg,
			"clustering": C,
			"average_path_length": L,
		},
		"erdos_renyi": {
			"p": p_er,
			"nodes": G_er.number_of_nodes(),
			"edges": G_er.number_of_edges(),
			"clustering": C_er,
			"average_path_length": L_er,
		},
		"barabasi_albert": {
			"m": M_INIT,
			"nodes": G_ba.number_of_nodes(),
			"edges": G_ba.number_of_edges(),
			"clustering": C_ba,
			"average_path_length": L_ba,
		},
		"watts_strogatz": {
			"k": k_ws,
			"p": p_ws,
			"nodes": G_ws.number_of_nodes(),
			"edges": G_ws.number_of_edges(),
			"clustering": C_ws,
			"average_path_length": L_ws,
		},
		"artifacts": {
			"plot": str(figure_path),
		},
	}

	json_path = out_path / f"{graph_name}_random_models_summary.json"
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=4)

	txt_path = out_path / f"{graph_name}_random_models_summary.txt"
	with open(txt_path, "w", encoding="utf-8") as f:
		f.write(f"Random Models Comparison for: {graph_name}\n")
		f.write("-" * 60 + "\n")
		f.write(f"Original Graph -> C: {C:.6f} | L: {L:.6f}\n")
		f.write(f"Erdos-Renyi   -> C: {C_er:.6f} | L: {L_er:.6f}\n")
		f.write(f"Barabasi-Albert -> C: {C_ba:.6f} | L: {L_ba:.6f}\n")
		f.write(f"Watts-Strogatz  -> C: {C_ws:.6f} | L: {L_ws:.6f}\n")
		f.write(f"Plot saved at: {figure_path}\n")
		f.write(f"JSON summary saved at: {json_path}\n")

	print(f"Report generation complete. Check the '{output_dir}' folder.")


if __name__ == "__main__":
	# Example usage:
	BASE_DIR = Path.cwd()
	input_file = BASE_DIR / "data" / "processed" / "world_trade_network_petrol_2024.gexf"

	if input_file.exists():
		random_graphs_comparison_report(
			graph_path=input_file,
			output_dir="results",
			graph_name="petrol_2024",
		)
	else:
		print(f"File not found: {input_file}. Please check the path.")
