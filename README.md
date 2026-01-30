# Cryptocurrency Fraud Detection via Graph Neural Networks

This repository implements a graph-based approach to cryptocurrency fraud detection using Graph Neural Networks (GNNs) on the Elliptic Bitcoin Transaction Dataset (~200k transactions). Transactions are modeled as nodes in a directed graph, with edges representing the flow of funds, enabling the use of relational information for illicit activity detection.

The project benchmarks multiple GNN architectures (GCN, GraphSAGE, GAT, GIN) against a Logistic Regression baseline. To prevent data leakage, a strict time-step–based train/test split is used, ensuring that future transactions are never used to predict past fraud. Extreme class imbalance is addressed using class-weighted loss functions.

GraphSAGE achieved the best trade-off between detection performance and training stability, significantly outperforming Logistic Regression and demonstrating the importance of graph structure in fraud detection.

Repository contents:
- gnn_random_split.ipynb: baseline experiment using random train/test splitting.
- gnn_time_series.ipynb: main experiment using strict temporal splitting (primary results).

Tech stack: Python, PyTorch, PyTorch Geometric (PyG), Scikit-learn.  

