import logging
import random
from dataclasses import dataclass

import numpy as np
import scipy as sp
from joblib import Parallel, delayed
from scipy.linalg import eig, inv
from sklearn.cluster import KMeans

from jointsbm.core.graph import GraphHandler
from jointsbm.utils.stats import estimateTheta, frobenius_norm, gen_design_matrix

from .models import SBMResults

logger = logging.getLogger(__name__)


def get_fac(Vn, V, K):
    x = np.eye(K)
    fac = np.nan_to_num(
        [
            [(Vn - x[i, :] + x[k, :]) / (V - x[i, :] + x[k, :]) for k in range(K)]
            for i in range(K)
        ]
    )
    # fac[fac<0] = 0
    gamman = fac.sum(axis=2)
    return fac, gamman


@dataclass
class SBMInitializer:
    graphs: GraphHandler
    K: int
    init_method = "kmeans++"
    seed: int = 32425066

    def initialize(self, parallel: bool = False):
        logger.info("Pulling pooled spectral embeddings...")

        if parallel:
            results = Parallel(n_jobs=-1)(
                delayed(self.process_single_graph)(graph) for graph in self.graphs
            )
        else:
            results = [self.process_single_graph(graph) for graph in self.graphs]
        self.X = [res[0] for res in results]
        self.Q = [res[1] for res in results]
        return self.X, self.Q

    def process_single_graph_spectral_embedding(self, graph):
        logger.debug("Processing single spectral embedding...")
        if sp.sparse.issparse(graph.adjacency_matrix):
            eig_decomp = sp.sparse.linalg.eigs(graph.adjacency_matrix, self.K)
        else:
            eig_decomp = eig(graph.adjacency_matrix)
        args = np.argsort(
            -abs(eig_decomp[0]),
        )[: self.K]
        D = eig_decomp[0][args]
        U = eig_decomp[1][:, args]
        return abs(np.matmul(U, np.diag(D)))

    def process_single_graph(self, graph):
        logger.debug("Processing single membership...")
        K = self.K
        Qn = self.process_single_graph_spectral_embedding(graph)
        n_nodes = graph.n_nodes
        if self.init_method == "kmeans++":
            km = KMeans(n_clusters=K, random_state=self.seed).fit(Qn).labels_
            Xn = gen_design_matrix(km)
        else:
            random.seed(self.seed)
            Xn = np.random.multinomial(1, [1.0 / K] * K, size=n_nodes)
        return Xn, Qn

    def __call__(self, *args, **kwds):
        return self.initialize(*args, **kwds)


@dataclass
class SBMEstimator:
    graphs: GraphHandler
    K: int
    max_iter: int = 100
    tol: float = 1e-25
    maxIter: int = 1000
    parallel: bool = False
    seed: int = 32425066

    def __post_init__(self):
        self.is_initialized = False

    def fit(self, return_soft_assignments=False):
        logger.info("Fitting SBM model...")

        if not self.is_initialized:
            logger.info("Model not initialized. Initializing now...")
            self.initialize()

        stopValue = np.inf
        oldLoss = np.inf
        iter = 0

        while stopValue > self.tol and iter < self.maxIter:
            iter += 1
            self.update_W()
            self.update_memberships()
            self.estimate_parameters()
            if self.cluster_sizes.min() == 0:
                logger.warning(
                    "One or more clusters have zero size. Reinitializing model."
                )
                self.is_initialized = False
                self.initialize(seed=random.randint(0, 100000))
            loss = self.compute_loss()
            if iter > 1:
                stopValue = abs(loss - oldLoss)
                if iter % 10 == 0:
                    logger.info(
                        f"Iteration {iter}: loss = {loss:.6f}, stopValue = {stopValue:.6f}"
                    )
                oldLoss = loss

        logger.info("Fitting complete.")
        logger.info(f"Final loss = {loss:.6f}, total iterations = {iter}")
        if return_soft_assignments:
            self.update_memberships(return_membership=False)

        theta = self.get_connectivity_matrix(allow_self_loops=False)
        order = np.argsort(
            -1 * np.diag(theta),
        )
        theta = theta[order,][:, order]
        self.theta = theta
        self.X = [self.X[i][:, order] for i in range(len(self.X))]

        memberships = {g.name: self.X[i] for i, g in enumerate(self.graphs)}

        return SBMResults(memberships=memberships, theta=theta)

    def initialize(self, seed=None):
        if self.is_initialized:
            logger.info("SBM model is already initialized.")
            return
        logger.info("Initializing SBM model...")
        seed = self.seed if seed is None else seed
        self.X, self.pooledQ = SBMInitializer(
            self.graphs, self.K, seed=seed
        ).initialize(parallel=False)
        self.estimate_parameters()
        self.is_initialized = True
        logger.info("Initialization complete.")

    def compute_loss(self):
        logger.debug("Computing loss...")
        loss = 0
        for graph_number in range(len(self.pooledQ)):
            XW = self.X[graph_number] @ self.W.T
            rt = self.rt_n[graph_number]
            term = np.sqrt(inv(self.delta2) @ self.delta2n[graph_number]) + np.sqrt(
                np.eye(self.K) * 1.0 / rt
            )
            loss += (
                self.gamma_n[graph_number]
                * frobenius_norm(XW - self.pooledQ[graph_number]) ** 2
                + frobenius_norm(np.matmul(self.pooledQ[graph_number], term)) ** 2
            )
        return loss

    def estimate_parameters(self):
        logger.debug("Estimating SBM parameters...")
        delta2n = []
        for x in self.X:
            delta2n.append((x.T @ x))
        self.delta2n = np.array(delta2n)
        self.delta2 = self.delta2n.sum(axis=0)
        self.cluster_sizes_n = (self.delta2n).diagonal(axis1=1, axis2=2)
        self.cluster_sizes = self.delta2.diagonal()
        self.gamma_n = 1 / self.cluster_sizes @ self.cluster_sizes_n.T
        self.rt_n = self.cluster_sizes.sum() / self.cluster_sizes_n.sum(axis=1)

    def update_W(self):
        logger.debug("Updating W parameter...")
        XnQn = []
        XXn = []
        for graph_number in range(len(self.pooledQ)):
            XnQn.append(
                self.X[graph_number].T
                @ (self.pooledQ[graph_number] * self.gamma_n[graph_number])
            )
            XXn.append(self.delta2n[graph_number] * self.gamma_n[graph_number])
        self.W = inv(np.sum(XXn, axis=0)) @ np.sum(XnQn, axis=0)

    def update_memberships(self, return_membership=True):
        if self.parallel:
            _ = Parallel(n_jobs=-1)(
                delayed(self.update_single_graph_memberships)(
                    graph_number, return_membership=return_membership
                )
                for graph_number in range(self.graphs.n_graphs)
            )
        else:
            _ = [
                self.update_single_graph_memberships(
                    graph_number, return_membership=return_membership
                )
                for graph_number in range(self.graphs.n_graphs)
            ]

    def update_single_graph_memberships(self, graph_number, return_membership=True):
        logger.debug(f"Updating memberships for graph {graph_number}...")
        Vn = self.cluster_sizes_n[graph_number]
        V = self.cluster_sizes
        fac, gamman = get_fac(Vn, V, self.K)
        rt = self.rt_n[graph_number]

        for i in range(self.X[graph_number].shape[0]):
            current_membership = self.X[graph_number][i,].argmax()
            self.X[graph_number][i,] = self.update_single_node_membership(
                self.pooledQ[graph_number][i,],
                rt,
                fac[current_membership],
                gamman[current_membership],
                return_membership=return_membership,
            )
        return

    def update_single_node_membership(self, q, rt, fac, gamman, return_membership=True):
        dif = self.W - q
        term2 = ((q * np.sqrt(fac) + q * 1.0 / np.sqrt(rt)) ** 2).sum(axis=1)
        dist = np.diag(np.matmul(dif, dif.T)) * gamman + term2
        if return_membership:
            membership = np.zeros(self.K)
            membership[dist.argmin()] = 1
            return membership
        else:
            return self.get_membership_probabilities(dist)

    def get_connectivity_matrix(self, X=None, allow_self_loops=False):
        if X is None:
            X = self.X
        total = np.zeros((self.K, self.K))
        obs = np.zeros((self.K, self.K))
        for i, x in enumerate(X):
            n_nodes = x.shape[0]
            __full_matrix = np.ones((n_nodes, n_nodes))
            if allow_self_loops == False:
                __full_matrix -= np.eye(n_nodes)
            total += x.T @ __full_matrix @ x
            obs += x.T @ self.graphs.graphs[i].adjacency_matrix @ x

        return obs / total

    def predict(self):
        logger.info("Predicting community assignments...")
        # Predict community assignments
        pass

    @staticmethod
    def get_membership_probabilities(dist):
        prob = np.exp(-dist)
        prob = prob / prob.sum()
        return prob
