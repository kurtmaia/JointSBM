"""
Graph processing utilities for Joint Stochastic Block Model (SBM).

This module provides utilities for handling multiple graphs, converting between
edge lists and sparse adjacency matrices, and preparing graph data for joint
spectral clustering algorithms.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def reindex_nodes(edgelist: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Reindex node labels in an edgelist to consecutive integers starting from 0.

    This function takes an edgelist with arbitrary node labels and creates a mapping
    to consecutive integer indices, which is useful for creating adjacency matrices.

    Args:
        edgelist (np.ndarray): Edge list array of shape (n_edges, 2) or (n_edges, 3).
                              First two columns contain source and target node IDs.

    Returns:
        Tuple[np.ndarray, Dict[int, int]]:
            - Reindexed edgelist with consecutive integer node IDs
            - Dictionary mapping original node IDs to new consecutive indices

    Example:
        >>> edges = np.array([[10, 20], [20, 30], [30, 10]])
        >>> reindexed_edges, mapping = reindex_nodes(edges)
        >>> print(reindexed_edges)
        [[0 1]
         [1 2]
         [2 0]]
        >>> print(mapping)
        {10: 0, 20: 1, 30: 2}
    """
    unique_nodes = np.unique(edgelist[:, :2].flatten())
    n_unique = unique_nodes.shape[0]
    node_mapping = dict(zip(unique_nodes, range(n_unique)))

    # Vectorized reindexing for better performance
    source_nodes = [node_mapping[edgelist[i, 0]] for i in range(edgelist.shape[0])]
    target_nodes = [node_mapping[edgelist[i, 1]] for i in range(edgelist.shape[0])]

    reindexed_edgelist = np.vstack([source_nodes, target_nodes]).T

    return reindexed_edgelist, node_mapping


def edgelist_to_sparse_matrix(
    edgelist: np.ndarray,
    symmetric: bool = True,
    n_nodes: Optional[int] = None,
    reindex: bool = False,
) -> Tuple[csr_matrix, Optional[Dict[int, int]]]:
    """
    Convert an edge list to a sparse adjacency matrix.

    This function converts a list of edges to a compressed sparse row (CSR) matrix
    representation, which is memory-efficient for sparse graphs.

    Args:
        edgelist (np.ndarray): Edge list array of shape (n_edges, 2) or (n_edges, 3).
                              First two columns contain source and target node IDs.
        symmetric (bool, optional): If True, make the matrix symmetric by adding
                                   the transpose. Defaults to True.
        n_nodes (Optional[int], optional): Number of nodes in the graph. If None,
                                          inferred from the maximum node ID. Defaults to None.
        reindex (bool, optional): If True, reindex nodes to consecutive integers
                                 starting from 0. Defaults to False.

    Returns:
        Tuple[csr_matrix, Optional[Dict[int, int]]]:
            - Sparse adjacency matrix in CSR format
            - Dictionary mapping new indices to original node IDs (only if reindex=True)

    Raises:
        ValueError: If edgelist is empty or has invalid shape.

    Example:
        >>> edges = np.array([[0, 1], [1, 2], [2, 0]])
        >>> adj_matrix, _ = edgelist_to_sparse_matrix(edges, symmetric=True)
        >>> print(adj_matrix.toarray())
        [[0 2 2]
         [2 0 2]
         [2 2 0]]
    """
    if edgelist.size == 0:
        raise ValueError("Edge list cannot be empty")

    if edgelist.shape[1] < 2:
        raise ValueError("Edge list must have at least 2 columns (source, target)")

    unique_nodes = np.unique(edgelist[:, :2].flatten())

    if n_nodes is None:
        n_nodes = int(max([unique_nodes.shape[0], unique_nodes.max() + 1]))

    reverse_mapping = None
    if reindex:
        edgelist, node_mapping = reindex_nodes(edgelist)
        # Create reverse mapping: new_index -> original_node_id
        reverse_mapping = {v: k for k, v in node_mapping.items()}

    # Create sparse matrix with edge weights (default to 1 if no weights provided)
    weights = np.ones(edgelist.shape[0])
    if edgelist.shape[1] > 2:  # If weights are provided
        weights = edgelist[:, 2]

    adjacency_matrix = csr_matrix(
        (weights, (edgelist[:, 0], edgelist[:, 1])), shape=(n_nodes, n_nodes)
    )

    if symmetric:
        if (adjacency_matrix != adjacency_matrix.T).nnz == 0:
            logger.debug("Adjacency matrix is already symmetric.")
        else:
            logger.debug("Making adjacency matrix symmetric.")
            adjacency_matrix = adjacency_matrix + adjacency_matrix.T

    return adjacency_matrix, reverse_mapping


@dataclass
class GraphObject:
    """
    A simple dataclass to encapsulate a graph's adjacency matrix and its name.

    Attributes:
        name (str): Name of the graph.
        adjacency_matrix (csr_matrix): Sparse adjacency matrix of the graph.
    """

    name: str
    adjacency_matrix: Union[csr_matrix, np.ndarray]
    n_nodes: int


@dataclass
class GraphHandler:
    """
    A handler class for processing and managing multiple graphs for Joint SBM analysis.

    This class provides a unified interface for handling single graphs, lists of graphs,
    or dictionaries of named graphs. It automatically converts edge lists to sparse
    adjacency matrices and provides utilities for graph preprocessing.

    Attributes:
        graph (Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]):
            Input graph(s) as edge lists or adjacency matrices.
        symmetric (bool): Whether to treat graphs as symmetric (undirected). Defaults to True.
        n_nodes (Optional[int]): Number of nodes in each graph. If None, inferred automatically.
        reindex (bool): Whether to reindex node labels to consecutive integers. Defaults to False.

    Properties (set after initialization):
        graphs (List[csr_matrix]): List of processed sparse adjacency matrices.
        graph_names (List[str]): List of graph names.
        n_graphs (int): Number of graphs.
        n_nodes_array (List[int]): Number of nodes in each graph.
        total_nodes (int): Total number of nodes across all graphs.
        idx2node (Optional[Dict[str, Dict[int, int]]]): Mapping from indices to original node IDs.

    Example:
        >>> # Single graph
        >>> edges = np.array([[0, 1], [1, 2], [2, 0]])
        >>> handler = GraphHandler(graph=edges)
        >>> print(f"Number of graphs: {handler.n_graphs}")

        >>> # Multiple graphs
        >>> graphs = [edges, edges]  # Two identical graphs
        >>> handler = GraphHandler(graph=graphs, symmetric=True)
        >>> print(f"Total nodes: {handler.total_nodes}")
    """

    graph: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]
    symmetric: bool = True
    n_nodes: Optional[int] = None
    reindex: bool = False

    def __post_init__(self):
        """
        Post-initialization processing to convert graphs to a standardized format.

        This method:
        1. Converts input graphs to a dictionary format
        2. Detects whether inputs are edge lists or adjacency matrices
        3. Converts edge lists to sparse adjacency matrices if needed
        4. Sets up graph metadata (names, counts, node counts)
        """
        # Convert input to dictionary format for consistent handling
        if isinstance(self.graph, list):
            self.graph = {f"graph_{i}": g for i, g in enumerate(self.graph)}
        elif not isinstance(self.graph, dict):
            self.graph = {"graph_0": self.graph}

        # Extract graph names
        self.graph_names = list(self.graph.keys())

        # Check if inputs are edge lists (non-square matrices) or adjacency matrices
        is_edgelist = False
        for graph_name, graph_data in self.graph.items():
            if graph_data.ndim != 2:
                raise ValueError(f"Graph {graph_name} must be a 2D array")
            if graph_data.shape[0] != graph_data.shape[1]:
                is_edgelist = True
                break

        if is_edgelist:
            logger.info("Converting edge lists to sparse adjacency matrices...")
            self.idx2node = {}
            processed_graphs = {}

            for graph_name in self.graph_names:
                logger.debug(f"Processing graph: {graph_name}")
                adj_matrix, reverse_mapping = edgelist_to_sparse_matrix(
                    self.graph[graph_name],
                    symmetric=self.symmetric,
                    n_nodes=self.n_nodes,
                    reindex=self.reindex,
                )
                processed_graphs[graph_name] = adj_matrix
                if reverse_mapping is not None:
                    self.idx2node[graph_name] = reverse_mapping
        else:
            # Already adjacency matrices, convert to sparse if needed
            processed_graphs = {}
            self.idx2node = None

            for graph_name, graph_data in self.graph.items():
                if not isinstance(graph_data, csr_matrix):
                    processed_graphs[graph_name] = csr_matrix(graph_data)
                else:
                    processed_graphs[graph_name] = graph_data

        # Store processed graphs and metadata
        self.graphs = [
            GraphObject(name, processed_graphs[name], processed_graphs[name].shape[0])
            for name in self.graph_names
        ]
        self.n_graphs = len(self.graphs)
        self.n_nodes_array = [graph.n_nodes for graph in self.graphs]
        self.total_nodes = sum(self.n_nodes_array)

        logger.info(
            f"Processed {self.n_graphs} graphs with {self.total_nodes} total nodes"
        )

    def get_graph_by_name(self, name: str) -> csr_matrix:
        """
        Retrieve a specific graph by name.

        Args:
            name (str): Name of the graph to retrieve.

        Returns:
            csr_matrix: The requested sparse adjacency matrix.

        Raises:
            KeyError: If the graph name is not found.
        """
        if name not in self.graph_names:
            raise KeyError(
                f"Graph '{name}' not found. Available graphs: {self.graph_names}"
            )

        index = self.graph_names.index(name)
        return self.graphs[index]

    def get_original_node_id(self, graph_name: str, node_index: int) -> int:
        """
        Get the original node ID for a given node index in a specific graph.

        This is only applicable if reindexing was performed during initialization.

        Args:
            graph_name (str): Name of the graph.
            node_index (int): Node index in the processed graph.

        Returns:
            int: Original node ID.

        Raises:
            ValueError: If reindexing was not performed or graph name is invalid.
        """
        if self.idx2node is None:
            raise ValueError("Node reindexing was not performed")

        if graph_name not in self.idx2node:
            raise ValueError(f"Graph '{graph_name}' not found in reindexing mapping")

        if node_index not in self.idx2node[graph_name]:
            raise ValueError(
                f"Node index {node_index} not found in graph '{graph_name}'"
            )

        return self.idx2node[graph_name][node_index]

    def get_summary(self) -> str:
        """
        Generate a summary string describing the graph collection.

        Returns:
            str: A formatted summary of the graphs.
        """
        summary_lines = [
            f"GraphHandler Summary:",
            f"  Number of graphs: {self.n_graphs}",
            f"  Graph names: {', '.join(self.graph_names)}",
            f"  Nodes per graph: {self.n_nodes_array}",
            f"  Total nodes: {self.total_nodes}",
            f"  Symmetric: {self.symmetric}",
            f"  Reindexed: {self.reindex}",
        ]

        return "\n".join(summary_lines)

    def summary(self) -> str:
        print(self.get_summary())

    def __iter__(self):
        """
        Allow iteration over processed graphs.

        Yields:
            csr_matrix: Each processed sparse adjacency matrix.
        """
        return iter(self.graphs)
