import logging

import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score as ari
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

logger = logging.getLogger(__name__)


def mcr(x, y):
    cm = contingency_matrix(x, y)
    return (cm.max(axis=0).sum()) * 1.0 / cm.sum()


class Evaluator:
    def __init__(self, groundTruth):
        self.groundTruth = groundTruth
        self.n_graphs = len(groundTruth)

    def evaluate(self, memberships):
        groundTruth = self.groundTruth
        n_graphs = self.n_graphs
        individual_nmi = {}
        individual_ari = {}
        individual_mcr = {}
        trueMemberships_stacked = []
        memberships_stacked = []
        for graph_name, membership in memberships.items():
            logger.debug(f"Evaluating graph {graph_name}...")
            individual_nmi[graph_name] = nmi(membership, groundTruth[graph_name])
            individual_ari[graph_name] = ari(membership, groundTruth[graph_name])
            individual_mcr[graph_name] = mcr(membership, groundTruth[graph_name])

            trueMemberships_stacked.append(groundTruth[graph_name])
            memberships_stacked.append(membership)

        trueMemberships_stacked = np.hstack(trueMemberships_stacked)
        memberships_stacked = np.hstack(memberships_stacked)
        overall_nmi = nmi(memberships_stacked, trueMemberships_stacked)
        overall_ari = ari(memberships_stacked, trueMemberships_stacked)
        overall_mcr = mcr(memberships_stacked, trueMemberships_stacked)

        return {
            "NMI": {
                "nmi": np.mean([measure for measure in individual_nmi.values()]),
                "overall_nmi": overall_nmi,
            },
            "ARI": {
                "ari": np.mean([measure for measure in individual_ari.values()]),
                "overall_ari": overall_ari,
            },
            "MCR": {
                "mcr": np.mean([measure for measure in individual_mcr.values()]),
                "overall_mcr": overall_mcr,
            },
        }
