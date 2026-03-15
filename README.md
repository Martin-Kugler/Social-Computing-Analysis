# Social Computing & Network Analysis

This repository contains a comprehensive social network analysis project for the **Social Computing and Personalization** course at the **Technical University of Madrid (UPM)**. The project focuses on leveraging advanced graph theory and GPU-accelerated computing to extract insights from real-world complex systems.


## Project Overview: World Trade Web (WTW) Analysis

The **World Trade Web** is a complex directed and weighted network where nodes represent sovereign nations and edges represent synchronized trade flows. Unlike simple graphs, the WTW exhibits a **"Rich-Club"** phenomenon and a core-periphery structure that defines the global economic hierarchy.

By modeling international trade using the **BACI dataset (HS92 Revision)**, this project aims to:

* **Map Global Dependencies**: Quantify the structural importance of nations beyond their GDP by using **Eigenvector Centrality** and **PageRank**.
* **Identify Trade Blocs**: Uncover hidden geopolitical clusters through **Louvain Community Detection**, revealing how geographic and political proximity influence economic integration.
* **Stress-Test the System**: Simulate "Targeted Attacks" on critical hubs (like major manufacturing or energy exporters) to measure the **cascading failures** and the overall decrease in global network efficiency.
* **Analyze Evolution**: Track the transition from a unipolar trade world to a multipolar system by comparing network snapshots across different decades.

The analysis includes:

* **Topology Characterization:** Global metrics (density, diameter, average path length).
* **Centrality Analysis:** Identification of influential nodes using Degree, Betweenness, and Eigenvector centralities.
* **Community Detection:** Clustering analysis using the Louvain Method to identify functional sub-groups.
* **Robustness Testing:** Evaluating network stability against targeted attacks on high-betweenness hubs.


## Technical Stack

* **Language:** Python 3.10
* **Core Library:** [NetworkX](https://networkx.org/)
* **GPU Acceleration:** [NVIDIA RAPIDS cuGraph](https://rapids.ai/libcugraph.html) (via NetworkX Dispatching)
* **Visualization:** Gephi (static) and PyVis (interactive HTML)
* **Environment Management:** Conda


## Hardware Acceleration (RTX 50-Series Blackwell)

This project is optimized for high-performance computing. It utilizes **NetworkX Dispatching** to offload heavy graph algorithms (like Betweenness Centrality) to the GPU.

* **GPU:** NVIDIA GeForce RTX 5050 (8GB VRAM)
* **Architecture:** Blackwell
* **CUDA Version:** 13.1
* **Backend:** `nx-cugraph`

> **Note on Reproducibility:** The code is designed to be hardware-agnostic. If a compatible NVIDIA GPU is not detected, NetworkX will automatically fallback to CPU execution without any code modifications.


## Getting Started

### Prerequisites
Ensure you have [Conda](https://docs.conda.io/en/latest/) installed on your system (**WSL2** is highly recommended for Windows users to ensure full CUDA compatibility).

### Data Setup
This project uses the **[BACI International Trade Database (HS92 Revision)](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37)**:

1) Download the raw CSV files for the desired years from CEPII BACI (click in the URL above).

2) Place the raw CSVs in data/raw/ and the country code dictionary in data/.

3) Run python src/data_processing.py to generate the optimized Parquet files.

### Installation

1) **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
   ```

2) **Create the environment from the environment.yml file:**

    ```bash
    conda env create -f environment.yml
    conda activate computacion_social
    ```

3) **(Optional) Enable GPU acceleration in your terminal:**

    ```bash
    export NETWORKX_AUTOMATIC_BACKENDS=cugraph
    ```

## Repository Structure
    data/               # Raw and processed datasets.
    notebooks/          # Exploratory Data Analysis (Jupyter Notebooks).
    results/            # Visualizations and final report.
    src/                # Production-ready Python scripts modularized for reusability.
    environment.yml     # Conda environment specification.
    README.md           # Project documentation.


## Project Roadmap & Navigation Guide

This section helps you understand the logical flow of the analysis and the purpose of each file in the repository.

### Execution Workflow (Notebooks and Scripts)

To replicate the study, notebooks and scripts should be executed in the following order:

1.  **`data_processing_pipeline_gist.ipynb`**: Initial data loading and cleaning pipeline (subsequently translated as a script in src as `data_processing.py` and its results exported to the `data/processed/` folder).

2.  **`graph_generation.ipynb`** and **`sepecific_product_graph_generation.ipynb`**: Graph generation pipeline from the processed data, as a general graph with all the products and as a graph with one specific product, respectively (subsequently translated as a script in src as `graph_generation.py` and its results exported to the `data/processed/` folder).

3.  **`graph_map_visualization.ipynb`**: The pipeline of the High-level geographic visualization of the resulting trade blocks (subsequently transalded as a script in src as `graph_map_visualization.py` and its results exported as a html to the `results/` folder).

4.  **`graph_metrics.ipynb`**: Basic analysis of the key metrics of the graph (subsequently translated as a script in src as `graph_metrics.py` and its results exports to the `results/` folder).

5.  **`graph_communities.ipynb`**: Implementation of community detection algorithms, such as Louvain, Girvan-Newman, and InfoMap (subsequently translated as a script in src as `graph_communities.py` and its results exported to the `results/` folder).

6.  **`random_graphs_comparison.ipynb`**: Comparative analysis between our real-world trade network and theoretical models (Erdős-Rényi, Watts-Strogatz, and Barabási-Albert) to prove non-random topology (subsequently translated as a script in src as `random_graphs_comparison.py` and its results exported to the `results/` folder).

7.  **`final_reporter.py`**: A master script to gather all metrics and generate a unified summary of the graph, exported to the `results/` folder.